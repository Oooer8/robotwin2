#!/bin/bash
# 检查所有任务的所有episode，三视角视频帧数与video帧数是否一致
# 依赖：ffprobe (ffmpeg)
# 使用方法
# 修改base_dir
# bash check_video_frames.sh

set -uo pipefail

base_dir="/root/workspace/robotwin_data/agilex"

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

# ── 颜色定义 ──────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ── 统计 ─────────────────────────────────────────────────
total_episodes=0
ok_episodes=0
mismatch_episodes=0
missing_episodes=0

# 用于汇总不一致的条目
mismatch_log=()

# ── 工具函数：获取视频帧数 ────────────────────────────────
get_frame_count() {
  local video_file="$1"
  if [ ! -f "${video_file}" ]; then
    echo "-1"
    return
  fi
  local count
  count=$(ffprobe -v error \
    -select_streams v:0 \
    -count_frames \
    -show_entries stream=nb_read_frames \
    -of csv=p=0 \
    "${video_file}" 2>/dev/null)
  # 如果 nb_read_frames 为空，fallback 到 nb_frames
  if [ -z "${count}" ] || [ "${count}" = "N/A" ]; then
    count=$(ffprobe -v error \
      -select_streams v:0 \
      -show_entries stream=nb_frames \
      -of csv=p=0 \
      "${video_file}" 2>/dev/null)
  fi
  echo "${count:-0}"
}

# ── 主循环 ───────────────────────────────────────────────
for task_name in "${tasks[@]}"; do
  collection_dir="${base_dir}/${task_name}/wm_agilex_100"
  replay_dir="${collection_dir}/robot_only_replay"
  video_dir="${collection_dir}/video"

  # 任务目录不存在则跳过
  if [ ! -d "${collection_dir}" ]; then
    echo -e "${YELLOW}[SKIP]${NC} 任务目录不存在: ${collection_dir}"
    continue
  fi

  echo -e "\n${CYAN}══════════════════════════════════════════${NC}"
  echo -e "${CYAN} 任务: ${task_name}${NC}"
  echo -e "${CYAN}══════════════════════════════════════════${NC}"

  # 找出所有 episode 目录
  shopt -s nullglob
  episode_dirs=("${replay_dir}"/episode*)
  shopt -u nullglob

  if [ ${#episode_dirs[@]} -eq 0 ]; then
    echo -e "${YELLOW}[SKIP]${NC} 无 episode 目录: ${replay_dir}"
    continue
  fi

  for ep_dir in "${episode_dirs[@]}"; do
    ep_name=$(basename "${ep_dir}")           # e.g. episode0
    ep_num="${ep_name#episode}"               # e.g. 0

    total_episodes=$((total_episodes + 1))

    # 三视角视频路径
    front_mp4="${ep_dir}/front.mp4"
    side_mp4="${ep_dir}/side.mp4"
    top_mp4="${ep_dir}/top.mp4"

    # 合并视频路径
    video_mp4="${video_dir}/${ep_name}.mp4"

    # 获取帧数
    front_frames=$(get_frame_count "${front_mp4}")
    side_frames=$(get_frame_count "${side_mp4}")
    top_frames=$(get_frame_count "${top_mp4}")
    video_frames=$(get_frame_count "${video_mp4}")

    # 检查文件是否缺失
    missing_files=()
    [ "${front_frames}" = "-1" ] && missing_files+=("front.mp4")
    [ "${side_frames}"  = "-1" ] && missing_files+=("side.mp4")
    [ "${top_frames}"   = "-1" ] && missing_files+=("top.mp4")
    [ "${video_frames}" = "-1" ] && missing_files+=("video/${ep_name}.mp4")

    if [ ${#missing_files[@]} -gt 0 ]; then
      echo -e "  ${YELLOW}[MISSING]${NC} ${ep_name}: 缺少文件 → ${missing_files[*]}"
      missing_episodes=$((missing_episodes + 1))
      continue
    fi

    # 判断帧数是否全部一致
    if [ "${front_frames}" = "${side_frames}" ] && \
       [ "${front_frames}" = "${top_frames}"  ] && \
       [ "${front_frames}" = "${video_frames}" ]; then
      echo -e "  ${GREEN}[OK]${NC} ${ep_name}: front=${front_frames} side=${side_frames} top=${top_frames} video=${video_frames}"
      ok_episodes=$((ok_episodes + 1))
    else
      echo -e "  ${RED}[MISMATCH]${NC} ${ep_name}: front=${front_frames} side=${side_frames} top=${top_frames} video=${video_frames}"
      mismatch_episodes=$((mismatch_episodes + 1))
      mismatch_log+=("${task_name}/${ep_name}: front=${front_frames} side=${side_frames} top=${top_frames} video=${video_frames}")
    fi
  done
done

# ── 汇总报告 ─────────────────────────────────────────────
echo -e "\n${CYAN}══════════════════════════════════════════${NC}"
echo -e "${CYAN} 检查完成 · 汇总报告${NC}"
echo -e "${CYAN}══════════════════════════════════════════${NC}"
echo -e " 总 episode 数  : ${total_episodes}"
echo -e " ${GREEN}帧数一致${NC}       : ${ok_episodes}"
echo -e " ${RED}帧数不一致${NC}     : ${mismatch_episodes}"
echo -e " ${YELLOW}文件缺失${NC}       : ${missing_episodes}"

if [ ${#mismatch_log[@]} -gt 0 ]; then
  echo -e "\n${RED}── 不一致详情 ──────────────────────────────${NC}"
  for entry in "${mismatch_log[@]}"; do
    echo -e "  ${RED}✗${NC} ${entry}"
  done
fi

echo -e "${CYAN}══════════════════════════════════════════${NC}\n"