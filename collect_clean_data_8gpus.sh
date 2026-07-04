#!/bin/bash
# ============================================================
# 收集 50 个任务、每个任务 50 条 clean 数据（无域随机化）。
# 使用 task_config/demo_clean.yml 配置：
#   - episode_num: 50          -> 每个任务 50 条
#   - domain_randomization 全部关闭, clean_background_rate: 1  -> clean 数据
#   - data_type.rgb: true      -> 自动导出 head/left/right 相机视频 (mp4)
#   - data_type.endpose: true  -> 保存 eef (末端位姿) 到 hdf5
#   - data_type.qpos: true     -> 保存关节动作到 hdf5
#   - collect_data: true       -> 自动生成 hdf5 + video
#
# 产物目录 (save_path 默认为 ./data):
#   data/<task_name>/demo_clean/data/episode{i}.hdf5              (含 eef/endpose)
#   data/<task_name>/demo_clean/video/episode{i}.mp4             (head 相机)
#   data/<task_name>/demo_clean/video/episode{i}_left_camera.mp4
#   data/<task_name>/demo_clean/video/episode{i}_right_camera.mp4
#
# 用法:
#   bash collect_clean_data_8gpus.sh [num_gpus]
#   # 例: bash collect_clean_data_8gpus.sh 8
# ============================================================

task_config="demo_clean"

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

# GPU 数量可通过第一个参数覆盖（默认 8）
num_gpus=${1:-8}

declare -A gpu_pid   # gpu_id -> 当前运行的 pid（空表示空闲）
task_idx=0
total=${#tasks[@]}

echo "使用配置: ${task_config}, GPU 数量: ${num_gpus}, 任务总数: ${total}"

# 初始阶段：填满所有 GPU
for (( gpu=0; gpu<num_gpus && task_idx<total; gpu++, task_idx++ )); do
  task="${tasks[$task_idx]}"
  echo "[$(date +%T)] 启动 GPU $gpu -> $task"
  bash collect_data.sh "$task" "$task_config" "$gpu" &
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
        bash collect_data.sh "$task" "$task_config" "$gpu" &
        gpu_pid[$gpu]=$!
        (( task_idx++ ))
      fi
    fi
  done
done

wait
echo "✅ 所有 ${total} 个任务的 clean 数据采集完成！"
