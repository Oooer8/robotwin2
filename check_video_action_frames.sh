#!/bin/bash
# ============================================================
# 检查 robotwin2_gtc_agilex_5k 数据集中：
#   4个视角视频帧数（head / urdf_front / urdf_side / urdf_top）
#   与 action JSON 中的 action_frames 是否一致
# 依赖: ffprobe (ffmpeg), python3 或 jq
# 用法: bash check_video_action_frames.sh
# ============================================================

set -uo pipefail

# BASE_DIR="${HOME}/workspace/final/dataset/robotwin2_gtc_agilex_5k_random"
BASE_DIR="/root/workspace/robotwin_data/robotwin2_gtc_agilex_pi0_3"

ACTION_DIR="${BASE_DIR}/action"
VIDEO_HEAD_DIR="${BASE_DIR}/videos/head"
VIDEO_FRONT_DIR="${BASE_DIR}/videos/urdf_front"
VIDEO_SIDE_DIR="${BASE_DIR}/videos/urdf_side"
VIDEO_TOP_DIR="${BASE_DIR}/videos/urdf_top"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

total=0
ok=0
mismatch=0
missing=0
mismatch_log=()
missing_log=()

get_frame_count() {
  local f="$1"
  if [ ! -f "${f}" ]; then
    echo "-1"
    return
  fi
  local count
  count=$(ffprobe -v error \
    -select_streams v:0 \
    -count_frames \
    -show_entries stream=nb_read_frames \
    -of csv=p=0 \
    "${f}" 2>/dev/null)
  if [ -z "${count}" ] || [ "${count}" = "N/A" ]; then
    count=$(ffprobe -v error \
      -select_streams v:0 \
      -show_entries stream=nb_frames \
      -of csv=p=0 \
      "${f}" 2>/dev/null)
  fi
  echo "${count:-0}"
}

get_action_frames() {
  local json_file="$1"
  if ! [ -f "${json_file}" ]; then
    echo "-1"
    return
  fi
  local val
  if command -v jq &>/dev/null; then
    val=$(jq -r '.action_frames // (.actions | length)' "${json_file}" 2>/dev/null)
  else
    val=$(python3 -c "
import json, sys
d = json.load(open('${json_file}'))
print(d.get('action_frames', len(d.get('actions', []))))
" 2>/dev/null)
  fi
  echo "${val:-0}"
}

if ! command -v ffprobe &>/dev/null; then
  echo -e "${RED}[ERROR]${NC} 未找到 ffprobe，请先安装 ffmpeg。"
  exit 1
fi

if [ ! -d "${ACTION_DIR}" ]; then
  echo -e "${RED}[ERROR]${NC} action 目录不存在: ${ACTION_DIR}"
  exit 1
fi

echo -e "${CYAN}${BOLD}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   robotwin2_gtc_agilex_5k 视频帧数 vs Action 一致性检查  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

while IFS= read -r -d '' json_file; do
  ep_id=$(basename "${json_file}" .json)

  total=$((total + 1))

  head_mp4="${VIDEO_HEAD_DIR}/${ep_id}.mp4"
  front_mp4="${VIDEO_FRONT_DIR}/${ep_id}_front.mp4"
  side_mp4="${VIDEO_SIDE_DIR}/${ep_id}_side.mp4"
  top_mp4="${VIDEO_TOP_DIR}/${ep_id}_top.mp4"

  action_frames=$(get_action_frames "${json_file}")
  head_frames=$(get_frame_count "${head_mp4}")
  front_frames=$(get_frame_count "${front_mp4}")
  side_frames=$(get_frame_count "${side_mp4}")
  top_frames=$(get_frame_count "${top_mp4}")

  missing_files=()
  [ "${head_frames}"   = "-1" ] && missing_files+=("head/${ep_id}.mp4")
  [ "${front_frames}"  = "-1" ] && missing_files+=("urdf_front/${ep_id}_front.mp4")
  [ "${side_frames}"   = "-1" ] && missing_files+=("urdf_side/${ep_id}_side.mp4")
  [ "${top_frames}"    = "-1" ] && missing_files+=("urdf_top/${ep_id}_top.mp4")
  [ "${action_frames}" = "-1" ] && missing_files+=("action/${ep_id}.json(读取失败)")

  if [ ${#missing_files[@]} -gt 0 ]; then
    echo -e "  ${YELLOW}[MISSING]${NC} ${ep_id}: 缺少 → ${missing_files[*]}"
    missing=$((missing + 1))
    missing_log+=("${ep_id}: 缺少 ${missing_files[*]}")
    continue
  fi

  if [ "${head_frames}"  = "${action_frames}" ] && \
     [ "${front_frames}" = "${action_frames}" ] && \
     [ "${side_frames}"  = "${action_frames}" ] && \
     [ "${top_frames}"   = "${action_frames}" ]; then
    echo -e "  ${GREEN}[OK]${NC}      ${ep_id}: action=${action_frames} | head=${head_frames} front=${front_frames} side=${side_frames} top=${top_frames}"
    ok=$((ok + 1))
  else
    echo -e "  ${RED}[MISMATCH]${NC} ${ep_id}: action=${action_frames} | head=${head_frames} front=${front_frames} side=${side_frames} top=${top_frames}"
    mismatch=$((mismatch + 1))
    mismatch_log+=("${ep_id}: action=${action_frames} head=${head_frames} front=${front_frames} side=${side_frames} top=${top_frames}")
  fi

done < <(find "${ACTION_DIR}" -maxdepth 1 -name "*.json" -print0 | sort -z)

# ── 汇总报告 ─────────────────────────────────────────────
echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}${BOLD}║             检查完成 · 汇总报告           ║${NC}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════╝${NC}"
printf " %-18s : %d\n" "总 episode 数" "${total}"
echo -e " ${GREEN}帧数全部一致${NC}     : ${ok}"
echo -e " ${RED}帧数不一致${NC}       : ${mismatch}"
echo -e " ${YELLOW}文件缺失${NC}         : ${missing}"

if [ ${#mismatch_log[@]} -gt 0 ]; then
  echo ""
  echo -e "${RED}${BOLD}── 不一致详情 ──────────────────────────────${NC}"
  for entry in "${mismatch_log[@]}"; do
    echo -e "  ${RED}✗${NC} ${entry}"
  done
fi

if [ ${#missing_log[@]} -gt 0 ]; then
  echo ""
  echo -e "${YELLOW}${BOLD}── 缺失文件详情 ────────────────────────────${NC}"
  for entry in "${missing_log[@]}"; do
    echo -e "  ${YELLOW}?${NC} ${entry}"
  done
fi

echo ""
[ $((mismatch + missing)) -eq 0 ] && exit 0 || exit 1