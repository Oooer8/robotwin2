#!/bin/bash
# 使用方法
# 1. 默认恢复 agilex_random 下 wm_agilex_100_random 的所有任务:
#    bash recover_all_episode_instructions.sh
# 2. 指定 task_config:
#    bash recover_all_episode_instructions.sh wm_agilex_100_random
# 3. 指定 task_config 和 base_dir:
#    bash recover_all_episode_instructions.sh wm_agilex_100_random /root/workspace/robotwin_data/agilex_random
# 4. 可选指定 max_num:
#    bash recover_all_episode_instructions.sh wm_agilex_100_random /root/workspace/robotwin_data/agilex_random 100
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_config="${1:-wm_agilex_100_random}"
base_dir="${2:-/root/workspace/robotwin_data/agilex_random}"
max_num="${3:-}"

recover_script="${script_dir}/recover_episode_instructions.py"
if [ ! -f "${recover_script}" ]; then
  echo "Cannot locate recover_episode_instructions.py: ${recover_script}" >&2
  exit 1
fi

if [ ! -d "${base_dir}" ]; then
  echo "Base dir not found: ${base_dir}" >&2
  exit 1
fi

max_num_args=()
if [ -n "${max_num}" ]; then
  max_num_args=(--max-num "${max_num}")
fi

success_count=0
skip_count=0
fail_count=0

shopt -s nullglob
task_dirs=("${base_dir}"/*)
shopt -u nullglob

if [ ${#task_dirs[@]} -eq 0 ]; then
  echo "No task directories found under ${base_dir}" >&2
  exit 1
fi

for task_dir in "${task_dirs[@]}"; do
  [ -d "${task_dir}" ] || continue

  task_name="$(basename "${task_dir}")"
  collection_dir="${task_dir}/${task_config}"
  scene_info_path="${collection_dir}/scene_info.json"

  echo "=========================================="
  echo "Task: ${task_name}"
  echo "Task config: ${task_config}"
  echo "Collection dir: ${collection_dir}"
  echo "=========================================="

  if [ ! -d "${collection_dir}" ]; then
    echo "[SKIP] missing collection dir: ${collection_dir}"
    skip_count=$((skip_count + 1))
    echo
    continue
  fi

  if [ ! -f "${scene_info_path}" ]; then
    echo "[SKIP] missing scene_info.json: ${scene_info_path}"
    skip_count=$((skip_count + 1))
    echo
    continue
  fi

  if python3 "${recover_script}" \
    --base-dir "${base_dir}" \
    --task-name "${task_name}" \
    --task-config "${task_config}" \
    "${max_num_args[@]}"; then
    success_count=$((success_count + 1))
    echo "[OK] recovered instructions for ${task_name}"
  else
    fail_count=$((fail_count + 1))
    echo "[FAILED] recover failed for ${task_name}"
  fi

  echo
done

echo "============== Summary =============="
echo "Task config: ${task_config}"
echo "Base dir: ${base_dir}"
echo "Recovered tasks: ${success_count}"
echo "Skipped tasks: ${skip_count}"
echo "Failed tasks: ${fail_count}"
echo "====================================="
