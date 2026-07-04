#!/usr/bin/env bash
# ============================================================
# 一体化脚本：
#   1) 只下载 HF 数据集里所有任务的 aloha-agilex_clean_50.zip
#   2) 解压并规整到失败采集所需的目录结构
#   3) 基于解压出的 clean 成功轨迹生成 clean 失败数据
#
# HF 数据集: https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset
#   dataset/<task>/aloha-agilex_clean_50.zip
#
# clean_50.zip 内含 (官方 pipeline 产物):
#   data/ (hdf5) video/ instructions/ _traj_data/ seed.txt scene_info.json
#   其中 _traj_data/ 和 seed.txt 是生成失败数据的必需源。
#
# 依赖: huggingface-cli (或 huggingface_hub[cli]) 、 unzip
#
# 用法:
#   bash download_clean50_and_collect_fail.sh
#   # 可用环境变量覆盖:
#   #   SAVE_PATH  解压/采集根目录 (默认 ./data)
#   #   RAW_ROOT   zip 下载缓存目录 (默认 ./raw_data)
#   #   NUM_GPUS   失败采集使用的 GPU 数 (默认 8)
#   #   MAX_JOBS   并行下载数 (默认 8)
#   #   RUN_FAIL   是否在准备完成后自动生成失败数据 (默认 true)
#   #   HF_ENDPOINT HF 镜像 (默认 https://hf-mirror.com)
#
# 例:
#   SAVE_PATH=/mnt/disk/robotwin_data NUM_GPUS=4 bash download_clean50_and_collect_fail.sh
# ============================================================

set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

REPO="TianxingChen/RoboTwin2.0"
CONFIG_ZIP="aloha-agilex_clean_50.zip"
COLLECTION_SUFFIX="aloha-agilex_clean_50"   # 解压后的目录名 = 失败采集的 source_collection_suffix

SAVE_PATH="${SAVE_PATH:-./data}"
RAW_ROOT="${RAW_ROOT:-./raw_data}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_JOBS="${MAX_JOBS:-8}"
RUN_FAIL="${RUN_FAIL:-true}"
FAIL_TASK_CONFIG="${FAIL_TASK_CONFIG:-demo_clean}"   # clean 场景配置 (无域随机化)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'

command -v unzip >/dev/null 2>&1 || { echo -e "${RED}[ERROR]${NC} 需要 unzip, 请先安装 (apt-get install -y unzip)"; exit 1; }

# HF CLI 兼容: 新版 huggingface_hub(1.x) 用 `hf`, 旧版用 `huggingface-cli`。
# 优先使用 `hf`, 因为新版里 `huggingface-cli` 已弃用为只打印提示的壳。
if command -v hf >/dev/null 2>&1; then
  HF_CLI_BIN="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI_BIN="huggingface-cli"
else
  echo -e "${RED}[ERROR]${NC} 未找到 hf / huggingface-cli, 请先安装: pip install -U 'huggingface_hub[cli]'"
  exit 1
fi

mkdir -p "$SAVE_PATH" "$RAW_ROOT"

# ---------- 步骤 1 + 2: 下载 + 解压 + 规整 ----------
prepare_one() {
  local task="$1"
  local zip_rel="dataset/${task}/${CONFIG_ZIP}"
  local zip_path="${RAW_ROOT}/${zip_rel}"
  local dest="${SAVE_PATH}/${task}/${COLLECTION_SUFFIX}"

  # 已准备好则跳过
  if [ -f "${dest}/seed.txt" ] && [ -d "${dest}/_traj_data" ]; then
    echo -e "${GREEN}[SKIP]${NC}  ${task} 已就绪"
    return 0
  fi

  # 下载 (仅这一个文件, 用位置参数指定文件路径, 跨版本最稳妥)
  if [ ! -f "$zip_path" ]; then
    echo -e "${YELLOW}[DL]${NC}    ${task}/${CONFIG_ZIP}"
    "$HF_CLI_BIN" download "$REPO" "$zip_rel" --repo-type dataset \
      --local-dir "$RAW_ROOT" || true
  fi
  if [ ! -f "$zip_path" ]; then
    echo -e "${RED}[FAIL]${NC}  ${task}: 下载后未找到 ${zip_path} (检查网络/HF_ENDPOINT/文件是否存在)"
    return 1
  fi

  # 解压到临时目录, 再规整到 dest
  local tmp; tmp="$(mktemp -d)"
  unzip -q -o "$zip_path" -d "$tmp"

  # 定位真正包含 _traj_data/seed.txt/data 的目录 (兼容 zip 内是否带顶层文件夹)
  local src=""
  if [ -e "$tmp/seed.txt" ] || [ -d "$tmp/_traj_data" ] || [ -d "$tmp/data" ]; then
    src="$tmp"
  else
    src="$(find "$tmp" -mindepth 1 -maxdepth 2 -type d \( -name _traj_data -o -name data \) -printf '%h\n' 2>/dev/null | head -n1)"
    if [ -z "$src" ]; then
      src="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d | head -n1)"
    fi
  fi

  if [ -z "$src" ] || [ ! -d "$src" ]; then
    echo -e "${RED}[FAIL]${NC}  ${task}: 解压后无法定位数据目录"
    rm -rf "$tmp"
    return 1
  fi

  mkdir -p "$dest"
  cp -a "$src"/. "$dest"/
  rm -rf "$tmp"

  # 校验失败采集必需的源
  if [ ! -f "${dest}/seed.txt" ] || [ ! -d "${dest}/_traj_data" ]; then
    echo -e "${RED}[WARN]${NC}  ${task}: 缺少 seed.txt 或 _traj_data/, 该任务无法生成失败数据"
    return 1
  fi
  echo -e "${GREEN}[DONE]${NC}  ${task} -> ${dest}"
}
export -f prepare_one
export RAW_ROOT SAVE_PATH REPO CONFIG_ZIP COLLECTION_SUFFIX GREEN RED YELLOW NC
export HF_ENDPOINT HF_CLI_BIN

echo "========================================"
echo " Repo        : $REPO"
echo " Config      : $CONFIG_ZIP"
echo " Save path   : $SAVE_PATH"
echo " Raw cache   : $RAW_ROOT"
echo " Tasks       : ${#tasks[@]}"
echo " Parallel DL : $MAX_JOBS"
echo "========================================"

printf '%s\n' "${tasks[@]}" \
  | xargs -P "$MAX_JOBS" -I {} bash -c 'prepare_one "$@"' _ {}

# ---------- 汇总准备结果 ----------
ready=0; missing=()
for task in "${tasks[@]}"; do
  dest="${SAVE_PATH}/${task}/${COLLECTION_SUFFIX}"
  if [ -f "${dest}/seed.txt" ] && [ -d "${dest}/_traj_data" ]; then
    ready=$((ready+1))
  else
    missing+=("$task")
  fi
done

echo "========================================"
echo -e "${GREEN}准备就绪任务: ${ready}/${#tasks[@]}${NC}"
if [ ${#missing[@]} -ne 0 ]; then
  echo -e "${RED}未就绪 (无法生成失败数据):${NC}"
  printf '  - %s\n' "${missing[@]}"
fi
echo "========================================"

# ---------- 步骤 3: 生成失败数据 ----------
if [ "$RUN_FAIL" != "true" ]; then
  echo "RUN_FAIL=false, 已跳过失败数据生成。手动执行:"
  echo "  bash collect_clean_fail_data_8gpus.sh ${NUM_GPUS} ${COLLECTION_SUFFIX} ${SAVE_PATH}"
  exit 0
fi

if [ "$ready" -eq 0 ]; then
  echo -e "${RED}[ERROR]${NC} 没有任何任务就绪, 中止失败数据生成。"
  exit 1
fi

echo -e "${YELLOW}[FAIL COLLECT]${NC} 使用 ${FAIL_TASK_CONFIG} 配置, source=${COLLECTION_SUFFIX}, save_path=${SAVE_PATH}"
bash collect_fail_data_agilex_8gpus.sh \
  "$FAIL_TASK_CONFIG" \
  "$NUM_GPUS" \
  "$COLLECTION_SUFFIX" \
  "$SAVE_PATH"

echo -e "${GREEN}✅ 全流程完成。失败数据位于 ${SAVE_PATH}/<task>/${COLLECTION_SUFFIX}_fail/${NC}"
