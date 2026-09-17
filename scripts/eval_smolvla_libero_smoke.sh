#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${1:-outputs/eval/smolvla_libero_spatial_task0_$(date +%Y%m%d_%H%M%S)}"

cd "${project_root}"

lerobot-eval \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --policy.device=cuda \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.task_ids='[0]' \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --seed=1000 \
  --output_dir="${output_dir}"
