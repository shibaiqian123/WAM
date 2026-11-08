# SONIC/Dex1 dataset for native 31 -> 66 post-training

This preset uses 20 Hz recordings, 31 measured state values (29 named body joints,
left/right Dex1 positions), and 66 raw target values (64 SONIC v1.1 latent values,
left/right continuous Dex1 targets in motor output radians). It adds no Root state.

Run the converter on a new output directory. It respects discarded episodes,
rejects whole episodes with any invalid Dex1 feedback/command/trigger flags, and
fails on non-finite numeric labels, invalid frame grids, or video counts.
Nearest timestamp sampling selects a whole source row and the corresponding video
frame. Half-frame ties within 2 microseconds choose the earlier row; the maximum
allowed 50-to-20 Hz sampling error is 10.01 milliseconds. No action shift or
latent interpolation is performed. Videos retain their original resolution and
are encoded with lossless H.264. The training loader resizes them to 224x224.

From the repository root:
```bash
python training/scripts/convert_sonic_dex1_dataset.py --source "$SOURCE" --audit-only
python training/scripts/convert_sonic_dex1_dataset.py --source "$SOURCE" --output "$OUTPUT" --workers 4
PYTHONPATH=training/src python training/scripts/validate_sonic_dex1_dataset.py --source "$SOURCE" --root "$OUTPUT"
```

From training/, using the training environment:
```bash
export WBWAM_SONIC_DATA_ROOT="$OUTPUT"
source scripts/load_env.sh
python scripts/compute_wb_norm_stats.py --output-dir "$OUTPUT/meta" wb_task=posttrain/sonic_dex1_20hz
python scripts/precompute_text_embeds.py wb_task=posttrain/sonic_dex1_20hz
python scripts/validate_sonic_dex1_dataset.py --root "$OUTPUT" --batch-only
```

The task preset reads `WBWAM_MIDTRAIN_CHECKPOINT` for weight initialization, not
resume. Set it to the downloaded physical midtrain checkpoint. The generated
data can only be used with the native SONIC model, not the original 96-D head.
Do not launch training until `.INCOMPLETE` is absent and both loader/normalized
batch checks pass. Training is a separate operation.

The fixed chronological 90/10 split is by entire episode (93 accepted episodes
give 83 train and 10 validation). Stats use only the training split, min/max
normalization, and the official stepwise 32-action statistics. Keep the supplied
stats and text cache with the dataset and preserve the task configuration.
The sample has 32 actions at 20 Hz and 9 video frames sampled every 4 steps.
The files include source episode/frame/timestamp provenance and reports for
exclusions, sampling, exact numeric equality, decoded RGB hashes and video PTS.
Lossless video preserves the decoded source frames, not the original compressed
bitstream. Dataset compatibility is not evidence of model convergence or robot
task success. During deployment the 20 Hz action timeline must not be played at
50 Hz without an explicit timing adapter.
