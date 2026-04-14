#!/bin/bash
# ============================================================
# collect_fail_data_agilex_8gpus.sh
#
# Parallel noisy-failure collection across all tasks that already
# have successful trajectory data.
#
# Expected source layout:
#   <save_path>/<task_name>/<source_collection_suffix>/_traj_data/episode*.pkl
#
# Usage:
#   bash collect_fail_data_agilex_8gpus.sh [task_config] [num_gpus] [source_collection_suffix] [save_path] [extra fail args...]
#
# Examples:
#   bash collect_fail_data_agilex_8gpus.sh wm_agilex_100 8 aloha-agilex_clean_50 /root/workspace/robotwin_data/robotwin/dataset
#   bash collect_fail_data_agilex_8gpus.sh demo_clean 8 aloha-agilex_clean_50 /path/to/robotwin_data --target-collection-suffix aloha-agilex_clean_50_fail_v2
#
# Notes:
#   - If save_path is omitted, task_config/<task_config>.yml is used.
#   - source_collection_suffix defaults to task_config.
#   - target collection defaults to <source_collection_suffix>_fail.
#   - Set PYTHON=/path/to/python if your environment does not expose "python".
# ============================================================

set -uo pipefail

task_config="${1:-wm_agilex_100}"
num_gpus="${2:-8}"
source_collection_suffix="${3:-$task_config}"
save_path_override="${4:-}"
extra_fail_args=("${@:5}")
python_bin="${PYTHON:-python}"

if ! [[ "$num_gpus" =~ ^[0-9]+$ ]] || (( num_gpus < 1 )); then
  echo "[ERROR] num_gpus must be a positive integer, got: $num_gpus"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

tmp_tasks="$(mktemp)"
if ! "$python_bin" - "$SCRIPT_DIR" "$task_config" "$source_collection_suffix" "$save_path_override" > "$tmp_tasks" <<'PY'
import sys
from pathlib import Path

import yaml

repo_root = Path(sys.argv[1]).resolve()
task_config = sys.argv[2]
source_collection_suffix = sys.argv[3]
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
    collection_dir = task_dir / source_collection_suffix
    traj_dir = collection_dir / "_traj_data"
    seed_path = collection_dir / "seed.txt"
    if seed_path.exists() and traj_dir.exists() and any(traj_dir.glob("episode*.pkl")):
        tasks.append(task_dir.name)

if not tasks:
    raise SystemExit(
        "[ERROR] No tasks found. Expected layout: "
        f"{save_path}/<task_name>/{source_collection_suffix}/_traj_data/episode*.pkl "
        "with seed.txt in the same collection directory."
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
LOG_DIR="logs/collect_fail/${task_config}_${source_collection_suffix}"
mkdir -p "$LOG_DIR"

echo "[INFO] task_config=$task_config"
echo "[INFO] source_collection_suffix=$source_collection_suffix"
echo "[INFO] target_collection=default ${source_collection_suffix}_fail unless overridden"
echo "[INFO] num_gpus=$num_gpus"
echo "[INFO] tasks=$total"
echo "[INFO] logs=$LOG_DIR"

next_idx=0
failed=0
running_pids=()
running_tasks=()

launch_task() {
  local gpu=$1
  local task=$2
  local log_file="$LOG_DIR/${task}.log"

  echo "[$(date +%T)] GPU $gpu -> $task (log: $log_file)"
  bash collect_fail_data.sh \
      "$task" \
      "$task_config" \
      "$gpu" \
      "$source_collection_suffix" \
      "$save_path_override" \
      "${extra_fail_args[@]}" \
      > "$log_file" 2>&1 &

  running_pids[$gpu]=$!
  running_tasks[$gpu]=$task
}

handle_done() {
  local gpu=$1
  local pid="${running_pids[$gpu]}"
  local task="${running_tasks[$gpu]}"
  local log_file="$LOG_DIR/${task}.log"
  local exit_code

  wait "$pid"
  exit_code=$?
  running_pids[$gpu]=""
  running_tasks[$gpu]=""

  if [[ $exit_code -eq 0 ]]; then
    echo "[$(date +%T)] GPU $gpu done: $task"
  else
    echo "[$(date +%T)] GPU $gpu failed(exit=$exit_code): $task -> $log_file"
    failed=1
  fi

  (( active_count-- ))
  if (( next_idx < total )); then
    launch_task "$gpu" "${tasks[$next_idx]}"
    (( next_idx++ ))
    (( active_count++ ))
  fi
}

active_count=0
for (( gpu=0; gpu<num_gpus && next_idx<total; gpu++ )); do
  launch_task "$gpu" "${tasks[$next_idx]}"
  (( next_idx++ ))
  (( active_count++ ))
done

while (( active_count > 0 )); do
  made_progress=0
  for (( gpu=0; gpu<num_gpus; gpu++ )); do
    pid="${running_pids[$gpu]:-}"
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      handle_done "$gpu"
      made_progress=1
    fi
  done

  if (( made_progress == 0 )); then
    sleep 2
  fi
done

if [[ $failed -eq 0 ]]; then
  echo "[INFO] All ${total} tasks finished."
else
  echo "[ERROR] Some tasks failed. Check logs under $LOG_DIR"
  exit 1
fi
