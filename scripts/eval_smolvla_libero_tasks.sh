#!/usr/bin/env bash

set -uo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_env="${FAILURE_AWARE_CONDA_ENV:-/home/sangpil/miniforge3/envs/failure-aware-vla}"
eval_bin="${conda_env}/bin/lerobot-eval"
run_name="${RUN_NAME:-spatial_tasks_8_9_10_seed1000_$(date +%Y%m%d_%H%M%S)}"
run_root="${project_root}/outputs/eval/${run_name}"
log_root="${run_root}/logs"

# Arguments are human-facing task numbers 1..10. LIBERO uses zero-based IDs 0..9.
if (( $# == 0 )); then
  task_numbers=(8 9 10)
else
  task_numbers=("$@")
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONUNBUFFERED=1

if [[ ! -x "${eval_bin}" ]]; then
  echo "lerobot-eval not found at ${eval_bin}" >&2
  exit 1
fi

mkdir -p "${log_root}"
printf 'task_number\ttask_id\tstatus\toutput_dir\tlog_file\n' > "${run_root}/status.tsv"

echo "Run directory: ${run_root}"
echo "Tasks (one-based): ${task_numbers[*]}"

overall_status=0
for task_number in "${task_numbers[@]}"; do
  if ! [[ "${task_number}" =~ ^([1-9]|10)$ ]]; then
    echo "Invalid task number: ${task_number} (expected 1..10)" >&2
    overall_status=1
    continue
  fi

  task_id=$((task_number - 1))
  task_label="task_$(printf '%02d' "${task_number}")_id_${task_id}"
  output_dir="${run_root}/${task_label}"
  log_file="${log_root}/${task_label}.log"

  printf '%s\t%s\tRUNNING\t%s\t%s\n' \
    "${task_number}" "${task_id}" "${output_dir}" "${log_file}" >> "${run_root}/status.tsv"
  echo "Starting task ${task_number} (LIBERO id ${task_id})"

  if "${eval_bin}" \
    --policy.path=HuggingFaceVLA/smolvla_libero \
    --policy.device=cuda \
    --policy.n_action_steps=10 \
    --env.type=libero \
    --env.task=libero_spatial \
    --env.task_ids="[${task_id}]" \
    --env.max_parallel_tasks=1 \
    --eval.batch_size=1 \
    --eval.n_episodes=10 \
    --seed=1000 \
    --output_dir="${output_dir}" 2>&1 | tee "${log_file}"; then
    status="DONE"
  else
    status="FAILED"
    overall_status=1
  fi

  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${task_number}" "${task_id}" "${status}" "${output_dir}" "${log_file}" >> "${run_root}/status.tsv"
  echo "Finished task ${task_number}: ${status}"
done

echo "All requested tasks processed. Status file: ${run_root}/status.tsv"
exit "${overall_status}"
