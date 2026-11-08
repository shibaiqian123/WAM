#!/usr/bin/env python3
"""Convert SONIC/Dex1 LeRobot v2.1 to a native 31/66 WB-WAM dataset.

Preserves source-row pairing. Nearest 20 Hz sampling uses the earlier row
on 10 ms ties. Re-encodes selected YUV420 frames losslessly, without resizing.
Writes a new directory only; source files are never modified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import av

BODY_NAMES = [
    "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint",
    "left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint",
    "right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
]
CAMERA = "observation.images.ego_view"
VALIDITY = ["dex1.feedback_valid", "dex1.command_valid", "dex1.trigger_valid"]

def load_json(path):
    return json.loads(path.read_text())

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

def matrix(table, key, dim):
    value = np.asarray(table[key].to_pylist(), dtype=np.float64)
    if value.shape != (len(table), dim):
        raise ValueError(f"{key}: invalid shape {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{key}: non-finite values")
    return value

def selection(timestamps):
    # Keep targets within the measured trajectory; never extrapolate.
    count = int(np.floor((timestamps[-1] - timestamps[0]) * 20 + 1e-4)) + 1
    targets = timestamps[0] + np.arange(count, dtype=np.float64) / 20
    right = np.searchsorted(timestamps, targets, side="left").clip(0, len(timestamps)-1)
    left = (right - 1).clip(0)
    # Source timestamps are float32; 2us tolerance stabilizes exact half-frame ties.
    idx = np.where(abs(timestamps[left]-targets) <= abs(timestamps[right]-targets)+2e-6, left, right)
    errors = timestamps[idx] - targets
    if np.any(np.diff(idx) <= 0) or np.max(abs(errors)) > 0.01001:
        raise ValueError("Non-unique sampling or >10.01ms timing error")
    return idx, targets, errors

def source_paths(root, info, ep):
    kw = dict(episode_index=ep, episode_chunk=ep//info["chunks_size"], video_key=CAMERA)
    return root/info["data_path"].format(**kw), root/info["video_path"].format(**kw)

def audit(root):
    info = load_json(root/"meta/info.json")
    if info["fps"] != 50:
        raise ValueError("This conversion requires 50Hz source recordings")
    names = info["features"]["observation.state"]["names"]
    body = [names.index(n) for n in BODY_NAMES]
    tasks = [json.loads(x) for x in (root/"meta/tasks.jsonl").read_text().splitlines() if x.strip()]
    eps = [json.loads(x) for x in (root/"meta/episodes.jsonl").read_text().splitlines() if x.strip()]
    discarded = set(info.get("discarded_episode_indices", []))
    accepted, excluded = [], []
    source_frames = 0
    for ep in eps:
        index = ep["episode_index"]
        path, video = source_paths(root, info, index)
        table = pq.read_table(path)
        source_frames += len(table)
        reasons = ["discarded"] if index in discarded else []
        for key in VALIDITY:
            v = matrix(table, key, 2)
            bad = int(np.count_nonzero(~np.all(v == 1, axis=1)))
            if bad: reasons.append(f"{key}:{bad}_invalid_frames")
        if reasons:
            excluded.append(dict(source_episode=index, reasons=reasons))
            continue
        state43 = matrix(table, "observation.state", 43)
        dex = matrix(table, "observation.dex1_position", 2)
        motion = matrix(table, "action.motion_token", 64)
        target = matrix(table, "action.dex1_target_position", 2)
        ts = np.asarray(table["timestamp"].to_pylist(),dtype=np.float64)
        if ts.shape != (len(table),) or not np.isfinite(ts).all() or np.any(np.diff(ts)<=0):
            raise ValueError(f"Invalid timestamps in {index}")
        if np.max(abs(ts-np.arange(len(table))/50)) > 1e-4:
            raise ValueError(f"Source video/frame grid is not 50Hz in {index}")
        if len(table)!=ep["length"] or not np.array_equal(table["frame_index"].to_numpy(),np.arange(len(table))):
            raise ValueError(f"Frame index/length mismatch in {index}")
        if set(table["episode_index"].to_pylist()) != {index} or len(set(table["task_index"].to_pylist())) != 1:
            raise ValueError(f"Episode/task mismatch in {index}")
        with av.open(str(video)) as reader:
            stream=reader.streams.video[0]
            if float(stream.average_rate)!=50 or stream.frames!=len(table):
                raise ValueError(f"Video metadata count/rate mismatch in {index}")
        idx, grid, errors = selection(ts)
        accepted.append(dict(source_episode=index, source_frames=len(table), frames=len(idx),
                             max_sampling_error_seconds=float(abs(errors).max()),
                             task_index=int(table["task_index"][0].as_py())))
    if source_frames != info["total_frames"] or len(eps)!=info["total_episodes"]:
        raise ValueError("Source totals mismatch")
    return info, body, tasks, accepted, excluded

def convert_one(job):
    root, out, info, body, entry, out_ep, global_start = job
    source_ep=entry["source_episode"]
    path,video=source_paths(root,info,source_ep)
    table=pq.read_table(path)
    ts=table["timestamp"].to_numpy().astype(np.float64)
    idx,targets,errors=selection(ts)
    states=np.concatenate([matrix(table,"observation.state",43)[:,body],
                           matrix(table,"observation.dex1_position",2)],axis=1)[idx].astype(np.float32)
    actions=np.concatenate([matrix(table,"action.motion_token",64),
                            matrix(table,"action.dex1_target_position",2)],axis=1)[idx].astype(np.float32)
    n=len(idx)
    chunk,file_index=out_ep//1000,out_ep%1000
    rel_data=f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet"
    rel_video=f"videos/{CAMERA}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
    dest=out/rel_video
    dest.parent.mkdir(parents=True,exist_ok=True)
    rgb_hash=hashlib.sha256()
    next_selected=0
    with av.open(str(video)) as reader, av.open(str(dest),"w") as writer:
        original=reader.streams.video[0]
        stream=writer.add_stream("libx264",rate=20)
        stream.width,stream.height=original.width,original.height
        stream.pix_fmt="yuv420p"
        stream.options={"crf":"0","preset":"fast"}
        stream.codec_context.thread_count=2
        decoded=0
        for i,frame in enumerate(reader.decode(video=0)):
            decoded+=1
            if next_selected>=n or i!=idx[next_selected]: continue
            if abs(float(frame.time)-ts[i])>1e-4: raise ValueError(f"Source video PTS mismatch: {source_ep}/{i}")
            if frame.format.name!="yuv420p": raise ValueError("Expected yuv420p source for lossless frame preservation")
            rgb_hash.update(frame.to_ndarray(format="rgb24").tobytes())
            frame.pts=next_selected
            frame.time_base=Fraction(1,20)
            for packet in stream.encode(frame): writer.mux(packet)
            next_selected+=1
        for packet in stream.encode(): writer.mux(packet)
        if decoded!=len(table) or next_selected!=n: raise ValueError("Decoded source count mismatch")
    data=pa.table({
        "observation.sonic_state":pa.array(states.tolist(),type=pa.list_(pa.float32(),31)),
        "action":pa.array(actions.tolist(),type=pa.list_(pa.float32(),66)),
        "timestamp":pa.array(np.arange(n,dtype=np.float64)/20),
        "frame_index":pa.array(np.arange(n,dtype=np.int64)),
        "episode_index":pa.array(np.full(n,out_ep,dtype=np.int64)),
        "index":pa.array(np.arange(global_start,global_start+n,dtype=np.int64)),
        "task_index":pa.array(np.full(n,entry["task_index"],dtype=np.int64)),
        "source_episode_index":pa.array(np.full(n,source_ep,dtype=np.int64)),
        "source_frame_index":pa.array(idx.astype(np.int64)),
        "source_timestamp":pa.array(ts[idx]),
        "sampling_error_seconds":pa.array(errors),
    })
    (out/rel_data).parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(data,out/rel_data,compression="zstd")
    row=dict(episode_index=out_ep,length=n,tasks=[entry["task_text"]],
             dataset_from_index=global_start,dataset_to_index=global_start+n,
             source_episode_index=source_ep)
    row.update({"data/chunk_index":chunk,"data/file_index":file_index,
                f"videos/{CAMERA}/chunk_index":chunk,f"videos/{CAMERA}/file_index":file_index,
                f"videos/{CAMERA}/from_timestamp":0.,f"videos/{CAMERA}/to_timestamp":n/20})
    report=dict(entry,output_episode=out_ep,data_path=rel_data,video_path=rel_video,
                source_selected_rgb_sha256=rgb_hash.hexdigest())
    print(f"Converted {out_ep}: source={source_ep}, {len(table)} -> {n}",flush=True)
    return row,report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--audit-only",action="store_true")
    args=parser.parse_args()
    root=args.source.resolve()
    info,body,tasks,accepted,excluded=audit(root)
    summary=dict(source_episodes=info["total_episodes"],accepted_episodes=len(accepted),
                 excluded=excluded,source_accepted_frames=sum(e["source_frames"] for e in accepted),
                 output_frames=sum(e["frames"] for e in accepted),source_body_indices=body)
    print(json.dumps(summary,indent=2),flush=True)
    if args.audit_only:return
    if args.output is None:parser.error("--output is required")
    out=args.output.resolve()
    if out.exists(): raise FileExistsError(f"Refusing existing output: {out}")
    out.mkdir(parents=True)
    (out/".INCOMPLETE").write_text("Conversion and validation pending\n")
    mapping={t["task_index"]:t["task"] for t in tasks}
    jobs=[];start=0
    for i,entry in enumerate(accepted):
        entry["task_text"]=mapping[entry["task_index"]]
        jobs.append((root,out,info,body,entry,i,start));start+=entry["frames"]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(convert_one,jobs))
    rows,reports=zip(*results)
    meta=out/"meta"; (meta/"episodes/chunk-000").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(list(rows)),meta/"episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist(tasks),meta/"tasks.parquet")
    features={
        "observation.sonic_state":{"dtype":"float32","shape":[31],"names":BODY_NAMES+["left_dex1_position","right_dex1_position"]},
        "action":{"dtype":"float32","shape":[66],"names":[f"sonic_latent_{i}" for i in range(64)]+["left_dex1_target","right_dex1_target"]},
        CAMERA:dict(info["features"][CAMERA]),
    }
    features[CAMERA]["info"]=dict(features[CAMERA].get("info",{}),**{"video.fps":20,"video.codec":"h264"})
    for key,typ in [("timestamp","float64"),("frame_index","int64"),("episode_index","int64"),("index","int64"),("task_index","int64"),
                    ("source_episode_index","int64"),("source_frame_index","int64"),("source_timestamp","float64"),("sampling_error_seconds","float64")]:
        features[key]={"dtype":typ,"shape":[1],"names":None}
    count=len(rows);train_count=max(1,int(count*.9))
    out_info=dict(codebase_version="v3.0",robot_type="unitree_g1_dex1",fps=20,total_episodes=count,
                  total_frames=start,total_tasks=len(tasks),total_videos=count,chunks_size=1000,
                  data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                  video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                  features=features,splits={"train":f"0:{train_count}","val":f"{train_count}:{count}"},
                  action_representation="sonic_latent64_dex1_continuous_v1",
                  source_script_config=info.get("script_config",{}))
    write_json(meta/"info.json",out_info)
    write_json(meta/"split_manifest.json",dict(method="chronological_whole_episode_90_10",
        train_episode_indices=list(range(train_count)),val_episode_indices=list(range(train_count,count)),
        train_source_episode_indices=[e["source_episode"] for e in reports[:train_count]],
        val_source_episode_indices=[e["source_episode"] for e in reports[train_count:]]))
    write_json(meta/"conversion_report.json",dict(summary,source=str(root),output=str(out),accepted=list(reports),
        source_fps=50,target_fps=20,resampling="nearest whole source row; ties within 2us choose earlier",
        video_encoding="libx264 lossless crf=0 yuv420p; original resolution; exact selected source RGB hashes",
        state_dim=31,action_dim=66,action_shift_frames=0,raw_values=True))
    print("CONVERSION COMPLETE; independent validation still required",flush=True)

if __name__=="__main__":main()
