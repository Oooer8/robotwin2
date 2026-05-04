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
collection_suffix="${3:-}"

tasks=(
  # adjust_bottle
  # beat_block_hammer
  blocks_ranking_rgb
  # blocks_ranking_size
  # click_alarmclock
  # click_bell
  # dump_bin_bigbin
  # grab_roller
  # handover_block
  # handover_mic
  # hanging_mug
  # lift_pot
  # move_can_pot
  # move_pillbottle_pad
  # move_playingcard_away
  # move_stapler_pad
  # open_laptop
  # open_microwave
  # pick_diverse_bottles
  # pick_dual_bottles
  # place_a2b_left
  # place_a2b_right
  # place_bread_basket
  # place_bread_skillet
  # place_burger_fries
  # place_can_basket
  # place_cans_plasticbox
  # place_container_plate
  # place_dual_shoes
  # place_empty_cup
  # place_fan
  # place_mouse_pad
  # place_object_basket
  # place_object_scale
  # place_object_stand
  # place_phone_stand
  # place_shoe
  # press_stapler
  # put_bottles_dustbin
  # put_object_cabinet
  # rotate_qrcode
  # scan_object
  # shake_bottle
  # shake_bottle_horizontally
  # stack_blocks_three
  # stack_blocks_two
  # stack_bowls_three
  # stack_bowls_two
  # stamp_seal
  # turn_switch
)

# tasks=(
#   place_phone_stand
#   place_mouse_pad
#   place_object_scale
#   press_stapler
#   place_object_stand
#   place_shoe
#   place_object_basket
#   rotate_qrcode
#   scan_object
#   place_cans_plasticbox
#   put_object_cabinet
#   shake_bottle
#   turn_switch
#   shake_bottle_horizontally
#   stack_bowls_two
#   put_bottles_dustbin
#   stack_bowls_three
#   stack_blocks_two
#   stack_blocks_three
#   stamp_seal
# )


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

next_idx=0
failed=0
running_pids=()
running_tasks=()

launch_task() {
  local gpu=$1
  local task=$2
  local log_file="$LOG_DIR/${task}.log"

  echo "[$(date +%T)] GPU $gpu -> $task (log: $log_file)"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" script/replay_robot_only.py \
      --task-name "$task" \
      --task-config "$task_config" \
      --collection-suffix "$collection_suffix" \
      --all-episodes \
      "${save_path_arg[@]}" \
      "${extra_replay_args[@]}" \
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
