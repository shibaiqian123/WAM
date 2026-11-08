#!/usr/bin/env python3
"""Independent full numeric/video audit plus real WB-WAM loader checks."""
import argparse,hashlib,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import av
import numpy as np
import pyarrow.parquet as pq
from wbwam.datasets.wb_dataset import WBDataset

BODY_NAMES=[
"left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
"right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
"waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
"left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
"right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"]
CAMERA="observation.images.ego_view"
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--source",type=Path);p.add_argument("--batch-only",action="store_true");a=p.parse_args()
 import torch
 torch.set_num_threads(4)
 r=a.root;s=a.source
 if a.batch_only:
  check_batch(r);return
 if s is None:p.error("--source is required for full validation")
 report=json.loads((r/"meta/conversion_report.json").read_text())
 info=json.loads((r/"meta/info.json").read_text())
 si=json.loads((s/"meta/info.json").read_text())
 body=[si["features"]["observation.state"]["names"].index(x) for x in BODY_NAMES]
 frames=0;max_error=0.;video_checks=[]
 def check_episode(e):
  ep=e["source_episode"];n=e["frames"]
  source=pq.read_table(s/si["data_path"].format(episode_chunk=ep//si["chunks_size"],episode_index=ep))
  target=pq.read_table(r/e["data_path"])
  idx=target["source_frame_index"].to_numpy()
  assert len(target)==n and np.all(np.diff(idx)>0)
  ts=source["timestamp"].to_numpy().astype(float)
  grid=np.arange(n)/20
  # Independently choose nearest source sample, with earlier exact-tie behavior.
  expected=np.array([int(np.flatnonzero(abs(ts-t)<=np.min(abs(ts-t))+2e-6)[0]) for t in grid])
  assert np.array_equal(idx,expected),(ep,"row selection")
  state=np.concatenate([np.asarray(source["observation.state"].to_pylist())[:,body],
                        np.asarray(source["observation.dex1_position"].to_pylist())],axis=1)[idx].astype(np.float32)
  action=np.concatenate([np.asarray(source["action.motion_token"].to_pylist()),
                         np.asarray(source["action.dex1_target_position"].to_pylist())],axis=1)[idx].astype(np.float32)
  assert np.array_equal(state,np.asarray(target["observation.sonic_state"].to_pylist(),dtype=np.float32))
  assert np.array_equal(action,np.asarray(target["action"].to_pylist(),dtype=np.float32))
  assert np.isfinite(state).all() and np.isfinite(action).all()
  assert np.array_equal(target["timestamp"].to_numpy(),grid)
  error=float(np.max(abs(ts[idx]-grid)));assert error<=0.01001
  h=hashlib.sha256();count=0
  with av.open(str(r/e["video_path"])) as reader:
   stream=reader.streams.video[0];assert float(stream.average_rate)==20
   for j,f in enumerate(reader.decode(video=0)):
    assert abs(float(f.time)-j/20)<1e-7
    h.update(f.to_ndarray(format="rgb24").tobytes());count+=1
  assert count==n and h.hexdigest()==e["source_selected_rgb_sha256"],(ep,"video content/count")
  print("VALIDATED",ep,n,flush=True)
  return n,error,e["output_episode"]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for n,error,output_ep in pool.map(check_episode,report["accepted"]):
   frames+=n;max_error=max(max_error,error);video_checks.append(output_ep)
 assert frames==info["total_frames"]
 kwargs=dict(root=str(r),dataset_format="official_lerobot_v3_1",state_key="observation.sonic_state",
 action_key="action",state_mask_key=None,action_mask_key=None,state_dim=31,action_dim=66,raw_state_dim=31,raw_action_dim=66,
 proprio_dim=31,action_target_dim=66,state_slices=None,action_slices=None,action_representation="native",
 camera_key=CAMERA,num_frames=33,action_video_freq_ratio=4,video_size=[224,224],val_set_proportion=.1,
 use_text_embed_cache=False,include_robot_metadata_in_prompt=False,strict_video_decode=True,video_backend="pyav")
 train=WBDataset(**kwargs,is_training_set=True);val=WBDataset(**kwargs,is_training_set=False)
 split=json.loads((r/"meta/split_manifest.json").read_text())
 assert [e["episode_index"] for e in train.episodes]==split["train_episode_indices"]
 assert [e["episode_index"] for e in val.episodes]==split["val_episode_indices"]
 sample_shapes=[]
 for ds in [train,val]:
  for index in [0,len(ds)//2,len(ds)-1]:
   x=ds[index];assert tuple(x["action"].shape)==(32,66)
   assert x["proprio"].shape[-1]==31
   assert tuple(x["video"].shape)==(3,9,224,224)
   if index==len(ds)-1: assert int(x["action_is_pad"].sum())==31
   sample_shapes.append({k:list(x[k].shape) for k in ["action","proprio","video"]})
 result=dict(passed=True,episodes=len(report["accepted"]),frames=frames,
 train_episodes=len(train.episodes),val_episodes=len(val.episodes),train_frames=len(train),val_frames=len(val),
 all_numeric_rows_match_source=True,all_video_rgb_hashes_match=True,all_video_pts_20hz=True,
 max_sampling_error_seconds=max_error,samples=sample_shapes,text_cache_checked=False,normalized_batch_checked=False)
 (r/"meta/validation_report.json").write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps(result,indent=2))

def check_batch(root):
    import os,torch
    from hydra import compose,initialize_config_dir
    from hydra.utils import instantiate
    from torch.utils.data._utils.collate import default_collate
    os.environ["WBWAM_SONIC_DATA_ROOT"]=str(root.resolve())
    # Only used to resolve the task config; this check never constructs a model.
    os.environ.setdefault("WBWAM_MIDTRAIN_CHECKPOINT","/unused/data-validation-only.pt")
    from wbwam.utils import misc
    misc.register_work_dir(str(root/"meta"))
    torch.set_num_threads(4)
    config_dir=Path(__file__).resolve().parents[1]/"configs"
    with initialize_config_dir(config_dir=str(config_dir),version_base="1.3"):
        cfg=compose(config_name="train",overrides=["wb_task=posttrain/sonic_dex1_20hz","data.image_augmentation.enabled=false"])
    train=instantiate(cfg.data,is_training_set=True)
    val=instantiate(cfg.data,is_training_set=False)
    report=json.loads((root/"meta/validation_report.json").read_text())
    cache=torch.load(next((root/"text_embeds_cache").glob("*.pt")),map_location="cpu",weights_only=True)
    assert int(cache["mask"].sum())==149
    max_roundtrip=0.
    for ds in [train,val]:
        child=ds.datasets[0]
        for index in [0,len(child)//2,len(child)-1]:
            sample=child[index]
            assert sample["context"].shape==(192,4096) and torch.isfinite(sample["context"]).all()
            # WB-WAM zeroes padded embeddings then exposes an all-valid mask.
            assert sample["context_mask"].sum().item()==192
            assert torch.count_nonzero(sample["context"][149:])==0
            ep,start=child._resolve_index(index)
            states,actions,_,_=child._read_sliced_episode_arrays(ep)
            indices=np.clip(np.arange(start,start+32),0,len(actions)-1)
            raw=torch.from_numpy(actions[indices])
            restored=ds.normalizer.denormalize_action(sample["action"])
            valid=~sample["action_is_pad"]
            err=float((restored[valid]-raw[valid]).abs().max())
            assert err<1e-4,err
            max_roundtrip=max(max_roundtrip,err)
        batch=default_collate([child[0],child[len(child)-1]])
        assert tuple(batch["action"].shape)==(2,32,66)
        assert batch["proprio"].shape[-1]==31
        assert tuple(batch["video"].shape)==(2,3,9,224,224)
        assert tuple(batch["context"].shape)==(2,192,4096)
        for key in ["action","proprio","video","context"]: assert torch.isfinite(batch[key]).all()
    report.update(text_cache_checked=True,normalized_batch_checked=True,
                  max_action_denormalization_error=max_roundtrip,
                  batch_shapes={k:list(batch[k].shape) for k in ["action","proprio","video","context","context_mask"]})
    (root/"meta/validation_report.json").write_text(json.dumps(report,indent=2)+"\n")
    (root/".INCOMPLETE").unlink(missing_ok=True)
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
