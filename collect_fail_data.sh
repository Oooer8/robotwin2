#!/bin/bash
# ============================================================
# collect_fail_data.sh
#
# Collect noisy failure data for one task from an existing
# successful trajectory collection.
#
# Expected source layout:
#   <save_path>/<task_name>/<source_collection_suffix>/_traj_data/episode*.pkl
#
# Usage:
#   bash collect_fail_data.sh <task_name> [task_config] [gpu_id] [source_collection_suffix] [save_path] [extra fail args...]
#
# Examples:
#   bash collect_fail_data.sh adjust_bottle wm_agilex_100 0 aloha-agilex_clean_50 /root/workspace/robotwin_data/robotwin/dataset
#   bash collect_fail_data.sh adjust_bottle wm_agilex_100 0 aloha-agilex_clean_50 /root/workspace/robotwin_data/robotwin/dataset --target-collection-suffix aloha-agilex_clean_50_fail_v2
#
# Notes:
#   - If save_path is omitted, task_config/<task_config>.yml is used.
#   - source_collection_suffix defaults to task_config.
#   - target collection defaults to <source_collection_suffix>_fail.
# ============================================================

set -uo pipefail

task_name=${1:?Usage: bash collect_fail_data.sh <task_name> [task_config] [gpu_id] [source_collection_suffix] [save_path] [extra fail args...]}
task_config=${2:-wm_agilex_100}
gpu_id=${3:-0}
source_collection_suffix=${4:-$task_config}
save_path_override=${5:-}
extra_fail_args=("${@:6}")
python_bin="${PYTHON:-python}"

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

save_path_arg=()
if [[ -n "$save_path_override" ]]; then
  save_path_arg=(--save-path "$save_path_override")
fi

PYTHONWARNINGS=ignore::UserWarning \
"$python_bin" script/collect_fail_data.py "$task_name" "$task_config" \
  --source-collection-suffix "$source_collection_suffix" \
  "${save_path_arg[@]}" \
  "${extra_fail_args[@]}"
