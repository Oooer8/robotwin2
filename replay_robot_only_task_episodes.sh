#!/bin/bash
# ============================================================
# replay_robot_only_task_episodes.sh
#
# Parallel robot-only replay across episodes for one task.
#
# Expected data layout:
#   <save_path>/<task_name>/<collection_suffix>/data/episode*.hdf5
#
# Usage:
#   bash replay_robot_only_task_episodes.sh <task_name> [task_config] [num_gpus] [collection_suffix] [save_path] [extra replay args...]
#
# Examples:
#   bash replay_robot_only_task_episodes.sh adjust_bottle demo_clean 8 aloha-agilex_clean_50 /path/to/robotwin_data
#   bash replay_robot_only_task_episodes.sh adjust_bottle demo_clean 8 aloha-agilex_clean_50 /path/to/robotwin_data --max-frames 30
#
# Notes:
#   - If save_path is omitted, the script uses save_path from task_config/<task_config>.yml.
#   - collection_suffix defaults to task_config.
#   - Set PYTHON=/path/to/python if your environment does not expose "python".
# ============================================================

set -uo pipefail

task_name="${1:-}"
task_config="${2:-demo_clean}"
num_gpus="${3:-8}"
collection_suffix="${4:-$task_config}"
save_path_override="${5:-}"
extra_replay_args=("${@:6}")
python_bin="${PYTHON:-python}"

if [[ -z "$task_name" ]]; then
  echo "Usage: bash $0 <task_name> [task_config] [num_gpus] [collection_suffix] [save_path] [extra replay args...]"
  echo "Example: bash $0 adjust_bottle demo_clean 8 aloha-agilex_clean_50 /path/to/robotwin_data"
  exit 1
fi

if ! [[ "$num_gpus" =~ ^[0-9]+$ ]] || (( num_gpus < 1 )); then
  echo "[ERROR] num_gpus must be a positive integer, got: $num_gpus"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

tmp_episodes="$(mktemp)"
if ! "$python_bin" - "$SCRIPT_DIR" "$task_config" "$task_name" "$collection_suffix" "$save_path_override" > "$tmp_episodes" <<'PY'
import sys
from pathlib import Path

import yaml

repo_root = Path(sys.argv[1]).resolve()
task_config = sys.argv[2]
task_name = sys.argv[3]
collection_suffix = sys.argv[4]
save_path_override = sys.argv[5]

cfg_path = repo_root / "task_config" / f"{task_config}.yml"
if not cfg_path.exists():
    raise SystemExit(f"[ERROR] Task config not found: {cfg_path}")

with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

save_path = Path(save_path_override) if save_path_override else Path(cfg.get("save_path", "./data"))
if not save_path.is_absolute():
    save_path = (repo_root / save_path).resolve()

collection_dir = save_path / task_name / collection_suffix
data_dir = collection_dir / "data"
if not data_dir.exists():
    raise SystemExit(f"[ERROR] Collected episode directory not found: {data_dir}")

episode_paths = sorted(
    data_dir.glob("episode*.hdf5"),
    key=lambda path: int(path.stem.replace("episode", "")),
)
if not episode_paths:
    raise SystemExit(f"[ERROR] No episode*.hdf5 files found in {data_dir}")

print(f"[INFO] collection_dir={collection_dir}", file=sys.stderr)
for path in episode_paths:
    print(path.stem.replace("episode", ""))
PY
then
  rm -f "$tmp_episodes"
  exit 1
fi

episodes=()
while IFS= read -r episode; do
  [[ -n "$episode" ]] && episodes+=("$episode")
done < "$tmp_episodes"
rm -f "$tmp_episodes"

total=${#episodes[@]}
LOG_DIR="logs/replay/${task_config}_${collection_suffix}_${task_name}_episodes"
mkdir -p "$LOG_DIR"

save_path_arg=()
if [[ -n "$save_path_override" ]]; then
  save_path_arg=(--save-path "$save_path_override")
fi

echo "[INFO] task_name=$task_name"
echo "[INFO] task_config=$task_config"
echo "[INFO] collection_suffix=$collection_suffix"
echo "[INFO] num_gpus=$num_gpus"
echo "[INFO] episodes=$total"
echo "[INFO] logs=$LOG_DIR"

worker() {
  local gpu=$1
  local failed=0
  local episode
  local log_file

  for (( idx=gpu; idx<total; idx+=num_gpus )); do
    episode="${episodes[$idx]}"
    log_file="$LOG_DIR/episode${episode}.log"

    echo "[$(date +%T)] GPU $gpu -> ${task_name}/episode${episode} (log: $log_file)"
    if CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" script/replay_robot_only.py \
      --task-name "$task_name" \
      --task-config "$task_config" \
      --collection-suffix "$collection_suffix" \
      --episode "$episode" \
      # --overwrite \
      "${save_path_arg[@]}" \
      "${extra_replay_args[@]}" \
      > "$log_file" 2>&1; then
      echo "[$(date +%T)] GPU $gpu done: ${task_name}/episode${episode}"
    else
      echo "[$(date +%T)] GPU $gpu failed: ${task_name}/episode${episode} -> $log_file"
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
  echo "[INFO] All ${total} episodes finished."
else
  echo "[ERROR] Some episodes failed. Check logs under $LOG_DIR"
  exit 1
fi
