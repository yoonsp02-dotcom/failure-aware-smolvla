# SmolVLA LIBERO smoke test — 2026-09-17

## Setup

- Policy: `HuggingFaceVLA/smolvla_libero`
- LeRobot commit: `d6039b8376a3ee099a6f0f64c1bb8599ca743ae3`
- Environment: `libero_spatial`, task 0
- Episodes: 1
- Seed: 1000
- Action execution horizon: 10
- Device: NVIDIA GeForce RTX 3060 Ti (8 GB)

## Result

- Success: 1/1
- Success rate: 100% (not statistically meaningful for one episode)
- Episode finished after 76 simulation steps
- Evaluation time: 37.40 seconds
- Video: `outputs/eval/smolvla_libero_spatial_task0_smoke_v2/videos/libero_spatial_0/eval_episode_0.mp4`
- Metrics: `outputs/eval/smolvla_libero_spatial_task0_smoke_v2/eval_info.json`

## Setup notes

- LIBERO uses its default paths stored in `~/.libero/config.yaml`.
- LIBERO assets are cached in `~/.cache/libero/assets`.
- No `rename_map` is needed with this LeRobot revision. The environment already
  emits `observation.images.image` and `observation.images.image2`, matching the
  checkpoint's expected inputs.
