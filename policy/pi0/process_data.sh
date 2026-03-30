#!/bin/bash
set -u

# ============================================================
# 批量把 RoboTwin 原始数据转换成 pi0 中间格式
# 需要先保证 ../../data/${task_name}/${task_config} 已经存在
#
# 用法:
#   bash process_data.sh
# ============================================================

task_config="wm_agilex_100"
expert_data_num=100
num_workers=25

tasks=(
  adjust_bottle
  # beat_block_hammer
  # blocks_ranking_rgb
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
  press_stapler
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

declare -A worker_pid   # worker_id -> pid
declare -A worker_task  # worker_id -> task

task_idx=0
total=${#tasks[@]}

echo "task_config=${task_config}"
echo "expert_data_num=${expert_data_num}"
echo "num_workers=${num_workers}"
echo "total_tasks=${total}"
echo

start_job() {
  local worker_id="$1"
  local task="$2"

  echo "[$(date +%T)] 启动 worker ${worker_id} -> ${task}"
  bash process_data_pi0.sh "$task" "$task_config" "$expert_data_num" &
  worker_pid[$worker_id]=$!
  worker_task[$worker_id]="$task"
}

# 初始填满 worker
for (( worker=0; worker<num_workers && task_idx<total; worker++, task_idx++ )); do
  start_job "$worker" "${tasks[$task_idx]}"
done

# 持续调度
while (( task_idx < total )); do
  wait -n

  for worker in "${!worker_pid[@]}"; do
    pid=${worker_pid[$worker]}
    if ! kill -0 "$pid" 2>/dev/null; then
      finished_task="${worker_task[$worker]}"
      echo "[$(date +%T)] worker ${worker} 完成 <- ${finished_task}"

      if (( task_idx < total )); then
        next_task="${tasks[$task_idx]}"
        start_job "$worker" "$next_task"
        (( task_idx++ ))
      else
        unset worker_pid[$worker]
        unset worker_task[$worker]
      fi
    fi
  done
done

# 等最后一批结束
for worker in "${!worker_pid[@]}"; do
  pid=${worker_pid[$worker]}
  task=${worker_task[$worker]}
  wait "$pid"
  echo "[$(date +%T)] worker ${worker} 完成 <- ${task}"
done

echo
echo "✅ 所有 ${total} 个任务处理完成！"
