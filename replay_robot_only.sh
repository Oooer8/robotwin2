#!/bin/bash
# ============================================================
# replay_robot_only.sh
#
# Parallel robot-only replay across all available tasks.
#
# Expected data layout:
#   <save_path>/<task_name>/<collection_suffix>/data/episode*.hdf5
#
# Usage:
#   bash replay_robot_only.sh [task_config] [num_gpus] [collection_suffix] [save_path] [extra replay args...]
#
# Examples:
#   bash replay_robot_only.sh demo_clean 8 aloha-agilex_clean_50 /path/to/robotwin_data
#   bash replay_robot_only.sh wm_agilex_100 8 wm_agilex_100_fail
#
# Notes:
#   - If save_path is omitted, the script uses save_path from task_config/<task_config>.yml.
#   - collection_suffix defaults to task_config.
#   - Set PYTHON=/path/to/python if your environment does not expose "python".
# ============================================================

set -uo pipefail

task_config="${1:-wm_agilex_100}"
num_gpus="${2:-8}"
collection_suffix="${3:-$task_config}"
save_path_override="${4:-}"
extra_replay_args=("${@:5}")
python_bin="${PYTHON:-python}"

if ! [[ "$num_gpus" =~ ^[0-9]+$ ]] || (( num_gpus < 1 )); then
  echo "[ERROR] num_gpus must be a positive integer, got: $num_gpus"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

tmp_tasks="$(mktemp)"
if ! "$python_bin" - "$SCRIPT_DIR" "$task_config" "$collection_suffix" "$save_path_override" > "$tmp_tasks" <<'PY'
import sys
from pathlib import Path

import yaml

repo_root = Path(sys.argv[1]).resolve()
task_config = sys.argv[2]
collection_suffix = sys.argv[3]
save_path_override = sys.argv[4]

cfg_path = repo_root / "task_config" / f"{task_config}.yml"
if not cfg_path.exists():
    raise SystemExit(f"[ERROR] Task config not found: {cfg_path}")

with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

save_path = Path(save_path_override) if save_path_override else Path(cfg.get("save_path", "./data"))
if not save_path.is_absolute():
    save_path = (repo_root / save_path).resolve()

if not save_path.exists():
    raise SystemExit(f"[ERROR] save_path does not exist: {save_path}")

tasks = []
for task_dir in sorted(path for path in save_path.iterdir() if path.is_dir()):
    data_dir = task_dir / collection_suffix / "data"
    if data_dir.exists() and any(data_dir.glob("episode*.hdf5")):
        tasks.append(task_dir.name)

if not tasks:
    raise SystemExit(
        "[ERROR] No tasks found. Expected layout: "
        f"{save_path}/<task_name>/{collection_suffix}/data/episode*.hdf5"
    )

print(f"[INFO] save_path={save_path}", file=sys.stderr)
for task in tasks:
    print(task)
PY
then
  rm -f "$tmp_tasks"
  exit 1
fi

tasks=()
while IFS= read -r task; do
  [[ -n "$task" ]] && tasks+=("$task")
done < "$tmp_tasks"
rm -f "$tmp_tasks"

total=${#tasks[@]}
LOG_DIR="logs/replay/${task_config}_${collection_suffix}"
mkdir -p "$LOG_DIR"

save_path_arg=()
if [[ -n "$save_path_override" ]]; then
  save_path_arg=(--save-path "$save_path_override")
fi

echo "[INFO] task_config=$task_config"
echo "[INFO] collection_suffix=$collection_suffix"
echo "[INFO] num_gpus=$num_gpus"
echo "[INFO] tasks=$total"
echo "[INFO] logs=$LOG_DIR"

worker() {
  local gpu=$1
  local failed=0
  local task
  local log_file

  for (( idx=gpu; idx<total; idx+=num_gpus )); do
    task="${tasks[$idx]}"
    log_file="$LOG_DIR/${task}.log"

    echo "[$(date +%T)] GPU $gpu -> $task (log: $log_file)"
    if CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" script/replay_robot_only.py \
      --task-name "$task" \
      --task-config "$task_config" \
      --collection-suffix "$collection_suffix" \
      --all-episodes \
      "${save_path_arg[@]}" \
      "${extra_replay_args[@]}" \
      > "$log_file" 2>&1; then
      echo "[$(date +%T)] GPU $gpu done: $task"
    else
      echo "[$(date +%T)] GPU $gpu failed: $task -> $log_file"
      failed=1
    fi
  done

  return "$failed"
}

pids=()
for (( gpu=0; gpu<num_gpus && gpu<total; gpu++ )); do
  worker "$gpu" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ $failed -eq 0 ]]; then
  echo "[INFO] All ${total} tasks finished."
else
  echo "[ERROR] Some tasks failed. Check logs under $LOG_DIR"
  exit 1
fi
