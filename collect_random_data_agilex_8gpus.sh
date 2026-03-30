#!/bin/bash
# ============================================================
# 要先修改task config
# bash collect_random_data_agilex_8gpus.sh 
# ============================================================

tasks=(
  adjust_bottle
  beat_block_hammer
  blocks_ranking_rgb
  blocks_ranking_size
  click_alarmclock
  click_bell
  dump_bin_bigbin
  grab_roller
  handover_block
  handover_mic
  hanging_mug
  lift_pot
  move_can_pot
  move_pillbottle_pad
  move_playingcard_away
  move_stapler_pad
  open_laptop
  open_microwave
  pick_diverse_bottles
  pick_dual_bottles
  place_a2b_left
  place_a2b_right
  place_bread_basket
  place_bread_skillet
  place_burger_fries
  place_can_basket
  place_cans_plasticbox
  place_container_plate
  place_dual_shoes
  place_empty_cup
  place_fan
  place_mouse_pad
  place_object_basket
  place_object_scale
  place_object_stand
  place_phone_stand
  place_shoe
  press_stapler
  put_bottles_dustbin
  put_object_cabinet
  rotate_qrcode
  scan_object
  shake_bottle
  shake_bottle_horizontally
  stack_blocks_three
  stack_blocks_two
  stack_bowls_three
  stack_bowls_two
  stamp_seal
  turn_switch
)


num_gpus=8

declare -A gpu_pid   # gpu_id -> 当前运行的 pid（空表示空闲）
task_idx=0
total=${#tasks[@]}

# 初始阶段：填满所有 GPU
for (( gpu=0; gpu<num_gpus && task_idx<total; gpu++, task_idx++ )); do
  task="${tasks[$task_idx]}"
  echo "[$(date +%T)] 启动 GPU $gpu -> $task"
  bash collect_random_data.sh "$task" wm_agilex_100_random "$gpu" &
  gpu_pid[$gpu]=$!
done

# 持续调度：有任务完成就立刻补充
while (( task_idx < total )); do
  wait -n  # 等待任意一个子进程结束

  # 找出已完成的 GPU
  for gpu in "${!gpu_pid[@]}"; do
    pid=${gpu_pid[$gpu]}
    if ! kill -0 "$pid" 2>/dev/null; then
      # 这个 GPU 空闲了，分配新任务
      if (( task_idx < total )); then
        task="${tasks[$task_idx]}"
        echo "[$(date +%T)] GPU $gpu 空闲，分配 -> $task"
        bash collect_random_data.sh "$task" wm_agilex_100_random "$gpu" &
        gpu_pid[$gpu]=$!
        (( task_idx++ ))
      fi
    fi
  done
done

wait
echo "✅ 所有 ${total} 个任务完成！"
