#!/bin/bash
# ============================================================
# replay_robot_only.sh
# 要先修改task config，确保路径正确
# 确保start seed注释掉
# 用法: bash replay_robot_only.sh [task_config] [num_gpus]

# 读 wm_agilex_100_fail 文件夹，但用 wm_agilex_100.yml 的配置
# bash replay_robot_only.sh wm_agilex_100 8 wm_agilex_100_fail
# 不传第三个参数，行为与原来完全一致（读 wm_agilex_100 文件夹）
# bash replay_robot_only.sh wm_agilex_100 8
# ============================================================

task_config="${1:-wm_agilex_100}"
num_gpus="${2:-8}"
collection_suffix="${3:-}"

tasks=(
  # adjust_bottle
  # beat_block_hammer
  # blocks_ranking_rgb
  # blocks_ranking_size
  # click_alarmclock
  # click_bell
  # dump_bin_bigbin
  grab_roller
  # handover_block
  # handover_mic
  hanging_mug
  # lift_pot
  # move_can_pot
  # move_pillbottle_pad
  # move_playingcard_away
  # move_stapler_pad
  # open_laptop
  open_microwave
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
  stack_bowls_three
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
task_idx=0

declare -A gpu_pid    # gpu_id -> pid
declare -A gpu_task   # gpu_id -> task name
declare -A pid_gpu    # pid   -> gpu_id  ← 新增反向映射，精确定位

LOG_DIR="logs/replay"
mkdir -p "$LOG_DIR"

# ── 启动单个任务 ────────────────────────────────────────────
launch_task() {
  local gpu=$1
  local task=$2
  local log_file="$LOG_DIR/${task}.log"

  local suffix_arg=""
  [[ -n "$collection_suffix" ]] && suffix_arg="--collection-suffix $collection_suffix"

  echo "[$(date +%T)] 🚀 GPU $gpu -> $task  (log: $log_file)"

  CUDA_VISIBLE_DEVICES="$gpu" \
    python script/replay_robot_only.py \
      --task-name   "$task" \
      --task-config "$task_config" \
      --all-episodes \
      $suffix_arg \
      > "$log_file" 2>&1 &

  local pid=$!
  gpu_pid[$gpu]=$pid
  gpu_task[$gpu]=$task
  pid_gpu[$pid]=$gpu   # 反向映射
}

# ── 处理一个已完成的 pid ────────────────────────────────────
handle_done() {
  local pid=$1
  local exit_code=$2
  local gpu=${pid_gpu[$pid]}
  local task=${gpu_task[$gpu]}

  unset pid_gpu[$pid]

  if [[ $exit_code -eq 0 ]]; then
    echo "[$(date +%T)] ✅ GPU $gpu 完成: $task"
  else
    echo "[$(date +%T)] ❌ GPU $gpu 失败(exit=$exit_code): $task -> 查看 $LOG_DIR/${task}.log"
  fi

  # 立刻补充新任务
  if (( task_idx < total )); then
    launch_task "$gpu" "${tasks[$task_idx]}"
    (( task_idx++ ))
  else
    unset gpu_pid[$gpu]
    unset gpu_task[$gpu]
  fi
}

# ── 阶段 1：填满所有 GPU ─────────────────────────────────────
for (( gpu=0; gpu<num_gpus && task_idx<total; gpu++, task_idx++ )); do
  launch_task "$gpu" "${tasks[$task_idx]}"
done

# ── 阶段 2：精确等待，完成一个立刻补一个 ──────────────────────
while (( task_idx < total )); do
  # wait -p 将结束的 pid 写入变量，-n 等待任意一个
  if wait -n -p finished_pid; then
    exit_code=0
  else
    exit_code=$?
  fi

  # finished_pid 需要 bash 5.1+；旧版本降级处理
  if [[ -n "${finished_pid:-}" && -n "${pid_gpu[$finished_pid]:-}" ]]; then
    handle_done "$finished_pid" "$exit_code"
  else
    # 降级：轮询找出已结束的进程
    for pid in "${!pid_gpu[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid"; ec=$?
        handle_done "$pid" "$ec"
      fi
    done
  fi
done

# ── 阶段 3：等待最后一批 ─────────────────────────────────────
echo "[$(date +%T)] ⏳ 等待最后 ${#gpu_pid[@]} 个任务完成..."
for gpu in "${!gpu_pid[@]}"; do
  pid=${gpu_pid[$gpu]}
  wait "$pid"; exit_code=$?
  task=${gpu_task[$gpu]}
  if [[ $exit_code -eq 0 ]]; then
    echo "[$(date +%T)] ✅ GPU $gpu 完成: $task"
  else
    echo "[$(date +%T)] ❌ GPU $gpu 失败(exit=$exit_code): $task -> 查看 $LOG_DIR/${task}.log"
  fi
done

echo ""
echo "🎉 所有 ${total} 个任务已处理完毕！"
echo "📁 日志目录: $LOG_DIR"