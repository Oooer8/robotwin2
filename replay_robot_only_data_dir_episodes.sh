#!/bin/bash
# ============================================================
# replay_robot_only_data_dir_episodes.sh
#
# Parallel robot-only replay across episode*.hdf5 files in a flat data directory.
#
# Expected data layout:
#   <data_dir>/episode*.hdf5
#
# Usage:
#   bash replay_robot_only_data_dir_episodes.sh <data_dir> [task_config] [num_gpus] [output_dir] [extra replay args...]
#
# Examples:
#   bash replay_robot_only_data_dir_episodes.sh \
#     /root/workspace/robotwin_data/robotwin/WorldArena_Robotwin2.0/data/fixed_scene_task \
#     wm_agilex_100 8 "" --overwrite
#
#   bash replay_robot_only_data_dir_episodes.sh \
#     /root/workspace/robotwin_data/robotwin/WorldArena_Robotwin2.0/data/fixed_scene_task \
#     wm_agilex_100 8 \
#     /root/workspace/robotwin_data/robotwin/WorldArena_Robotwin2.0/data/fixed_scene_task/robot_only_replay \
#     --max-frames 30 --overwrite
#
# Notes:
#   - Requires script/replay_robot_only.py to support --data-dir.
#   - Existing complete episodes are skipped by default. Pass --overwrite or --force-overwrite to regenerate them.
#   - Set PYTHON=/path/to/python if your environment does not expose "python".
# ============================================================

set -uo pipefail

data_dir="${1:-}"
task_config="${2:-wm_agilex_100}"
num_gpus="${3:-8}"
output_dir="${4:-}"
extra_replay_args=("${@:5}")
python_bin="${PYTHON:-python}"
overwrite_arg=()

if [[ -z "$data_dir" ]]; then
  echo "Usage: bash $0 <data_dir> [task_config] [num_gpus] [output_dir] [extra replay args...]"
  echo "Example: bash $0 /path/to/fixed_scene_task wm_agilex_100 8 \"\" --overwrite"
  exit 1
fi

if ! [[ "$num_gpus" =~ ^[0-9]+$ ]] || (( num_gpus < 1 )); then
  echo "[ERROR] num_gpus must be a positive integer, got: $num_gpus"
  exit 1
fi

filtered_extra_args=()
for arg in "${extra_replay_args[@]}"; do
  if [[ "$arg" == "--force-overwrite" ]]; then
    overwrite_arg=(--overwrite)
  else
    filtered_extra_args+=("$arg")
  fi
done
extra_replay_args=("${filtered_extra_args[@]}")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

tmp_episodes="$(mktemp)"
if ! "$python_bin" - "$data_dir" > "$tmp_episodes" <<'PY'
import re
import sys
from pathlib import Path

data_dir = Path(sys.argv[1]).expanduser().resolve()
if not data_dir.is_dir():
    raise SystemExit(f"[ERROR] data_dir does not exist or is not a directory: {data_dir}")

episode_paths = sorted(
    data_dir.glob("episode*.hdf5"),
    key=lambda path: (
        0,
        int(re.search(r"episode(\d+)", path.stem).group(1)),
    )
    if re.search(r"episode(\d+)", path.stem)
    else (1, path.stem),
)
if not episode_paths:
    raise SystemExit(f"[ERROR] No episode*.hdf5 files found in {data_dir}")

print(f"[INFO] data_dir={data_dir}", file=sys.stderr)
for path in episode_paths:
    match = re.search(r"episode(\d+)", path.stem)
    if match is None:
        continue
    print(match.group(1))
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
data_dir_abs="$("$python_bin" - "$data_dir" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]).expanduser().resolve())
PY
)"

LOG_DIR="logs/replay_data_dir/$(basename "$data_dir_abs")_${task_config}_episodes"
mkdir -p "$LOG_DIR"

output_dir_arg=()
if [[ -n "$output_dir" ]]; then
  output_dir_arg=(--output-dir "$output_dir")
fi

echo "[INFO] data_dir=$data_dir_abs"
echo "[INFO] task_config=$task_config"
echo "[INFO] num_gpus=$num_gpus"
echo "[INFO] episodes=$total"
echo "[INFO] output_dir=${output_dir:-${data_dir_abs}/robot_only_replay}"
echo "[INFO] logs=$LOG_DIR"

next_idx=0
failed=0
active_count=0
running_pids=()
running_episodes=()

launch_episode() {
  local gpu=$1
  local episode=$2
  local log_file="$LOG_DIR/episode${episode}.log"

  echo "[$(date +%T)] GPU $gpu -> episode${episode} (log: $log_file)"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" script/replay_robot_only.py \
      --task-config "$task_config" \
      --data-dir "$data_dir_abs" \
      --episode "$episode" \
      "${overwrite_arg[@]}" \
      "${output_dir_arg[@]}" \
      "${extra_replay_args[@]}" \
      > "$log_file" 2>&1 &

  running_pids[$gpu]=$!
  running_episodes[$gpu]=$episode
}

handle_done() {
  local gpu=$1
  local pid="${running_pids[$gpu]}"
  local episode="${running_episodes[$gpu]}"
  local log_file="$LOG_DIR/episode${episode}.log"
  local exit_code

  wait "$pid"
  exit_code=$?
  running_pids[$gpu]=""
  running_episodes[$gpu]=""

  if [[ $exit_code -eq 0 ]]; then
    echo "[$(date +%T)] GPU $gpu done: episode${episode}"
  else
    echo "[$(date +%T)] GPU $gpu failed(exit=$exit_code): episode${episode} -> $log_file"
    failed=1
  fi

  (( active_count-- ))
  if (( next_idx < total )); then
    launch_episode "$gpu" "${episodes[$next_idx]}"
    (( next_idx++ ))
    (( active_count++ ))
  fi
}

for (( gpu=0; gpu<num_gpus && next_idx<total; gpu++ )); do
  launch_episode "$gpu" "${episodes[$next_idx]}"
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
  echo "[INFO] All ${total} episodes finished."
else
  echo "[ERROR] Some episodes failed. Check logs under $LOG_DIR"
  exit 1
fi
