#!/bin/bash
# 导出所有任务 episode*.hdf5 中的 action JSON，并按 convert_dataset 的 key 规则命名。
#
# 新数据推荐用法：
#   bash export_all_action.sh \
#     --robotwin-data /root/workspace/robotwin_data \
#     --tasks-root robotwin/dataset \
#     --variant aloha-agilex_clean_50 \
#     --output-dir /root/workspace/robotwin_data/robotwin2_gtc_aloha_agilex_clean_50/action
#
# 兼容旧用法：
#   bash export_all_action.sh wm_agilex_100
#   bash export_all_action.sh wm_agilex_100 /root/workspace/robotwin_data/agilex_eval

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash export_all_action.sh [options]
  bash export_all_action.sh <variant_or_task_config> [collection_root]

Options:
  --robotwin-data PATH       robotwin_data 根目录。
  --tasks-root PATH          任务目录相对 robotwin_data 的路径，默认 robotwin/dataset。
  --variant NAME             任务下的数据子目录名，例如 aloha-agilex_clean_50。
  --task-config NAME         --variant 的别名，兼容旧命名。
  --base-dir PATH            直接指定包含任务文件夹的根目录，例如 .../robotwin/dataset。
  --collection-root PATH     --base-dir 的别名。
  --output-dir PATH          action JSON 输出目录。
  --tasks a,b,c              只导出指定任务，逗号分隔；编号仍按全局 TASK_ORDER。
  --tmp-dir PATH             临时 episode*_actions.json 输出目录；默认使用 mktemp。
  --overwrite                目标 JSON 已存在时覆盖。
  --no-overwrite             目标 JSON 已存在时跳过，默认行为。
  --dry-run                  只打印将要导出的文件，不实际写入。
  -h, --help                 显示帮助。

Examples:
  bash export_all_action.sh \
    --robotwin-data /root/workspace/robotwin_data \
    --variant aloha-agilex_clean_50 \
    --output-dir /root/workspace/robotwin_data/robotwin2_gtc_aloha_agilex_clean_50/action

  bash export_all_action.sh \
    --base-dir /root/workspace/robotwin_data/robotwin/dataset \
    --variant aloha-agilex_clean_50 \
    --tasks beat_block_hammer,lift_pot \
    --output-dir /root/workspace/action_clean_50
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "${script_dir}/task_config" ]; then
  repo_root="${script_dir}"
elif [ -d "${script_dir}/../task_config" ]; then
  repo_root="$(cd "${script_dir}/.." && pwd)"
else
  echo "Cannot locate repo root from ${script_dir}" >&2
  exit 1
fi

script_path="${repo_root}/export_actions_json.py"
if [ ! -f "${script_path}" ]; then
  echo "Cannot locate export_actions_json.py in repo root: ${repo_root}" >&2
  exit 1
fi

default_variant="wm_agilex_100"
variant=""
robotwin_data=""
tasks_root="robotwin/dataset"
base_dir_override=""
final_output_dir=""
tmp_output_dir=""
tasks_csv=""
dry_run=0
overwrite=0
cleanup_tmp=0
positional=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --robotwin-data|--input-root)
      robotwin_data="${2:?missing value for $1}"
      shift 2
      ;;
    --tasks-root|-t)
      tasks_root="${2:?missing value for $1}"
      shift 2
      ;;
    --variant|-v|--task-config)
      variant="${2:?missing value for $1}"
      shift 2
      ;;
    --base-dir|--collection-root)
      base_dir_override="${2:?missing value for $1}"
      shift 2
      ;;
    --output-dir|--output|-o)
      final_output_dir="${2:?missing value for $1}"
      shift 2
      ;;
    --tmp-dir)
      tmp_output_dir="${2:?missing value for $1}"
      shift 2
      ;;
    --tasks)
      tasks_csv="${2:?missing value for $1}"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --overwrite)
      overwrite=1
      shift
      ;;
    --no-overwrite)
      overwrite=0
      shift
      ;;
    --)
      shift
      while [ "$#" -gt 0 ]; do
        positional+=("$1")
        shift
      done
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      positional+=("$1")
      shift
      ;;
  esac
done

if [ "${#positional[@]}" -gt 0 ] && [ -z "${variant}" ]; then
  variant="${positional[0]}"
fi
if [ "${#positional[@]}" -gt 1 ] && [ -z "${base_dir_override}" ]; then
  base_dir_override="${positional[1]}"
fi
if [ "${#positional[@]}" -gt 2 ]; then
  echo "Too many positional arguments: ${positional[*]}" >&2
  usage >&2
  exit 2
fi

variant="${variant:-${default_variant}}"

resolve_path() {
  python3 - "$1" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
}

join_root_path() {
  python3 - "$1" "$2" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).expanduser()
child = Path(sys.argv[2])
print((child if child.is_absolute() else root / child).resolve())
PY
}

if [ -n "${robotwin_data}" ]; then
  robotwin_data="$(resolve_path "${robotwin_data}")"
fi

if [ -z "${base_dir_override}" ] && [ -n "${robotwin_data}" ]; then
  base_dir_override="$(join_root_path "${robotwin_data}" "${tasks_root}")"
elif [ -n "${base_dir_override}" ]; then
  base_dir_override="$(resolve_path "${base_dir_override}")"
fi

if [ -z "${final_output_dir}" ]; then
  if [ -n "${robotwin_data}" ]; then
    final_output_dir="${robotwin_data}/action"
  else
    # 兼容旧脚本默认输出。
    final_output_dir="/root/workspace/action"
  fi
fi
final_output_dir="$(resolve_path "${final_output_dir}")"

if [ -z "${tmp_output_dir}" ]; then
  if [ "${dry_run}" -eq 1 ]; then
    tmp_output_dir="${TMPDIR:-/tmp}/robotwin_action_tmp.DRYRUN"
    cleanup_tmp=0
  else
    tmp_output_dir="$(mktemp -d "${TMPDIR:-/tmp}/robotwin_action_tmp.XXXXXX")"
    cleanup_tmp=1
  fi
else
  tmp_output_dir="$(resolve_path "${tmp_output_dir}")"
  cleanup_tmp=0
fi

all_tasks=(
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

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

if [ -n "${tasks_csv}" ]; then
  tasks=()
  IFS=',' read -r -a raw_tasks <<< "${tasks_csv}"
  for raw_task in "${raw_tasks[@]}"; do
    task="$(trim "${raw_task}")"
    if [ -n "${task}" ]; then
      tasks+=("${task}")
    fi
  done
else
  tasks=("${all_tasks[@]}")
fi

task_id_for_name() {
  local task_name="$1"
  local i
  for ((i=0; i<${#all_tasks[@]}; i++)); do
    if [ "${all_tasks[i]}" = "${task_name}" ]; then
      printf "%03d" $((i + 1))
      return 0
    fi
  done
  return 1
}

resolve_collection_dir() {
  local task_name="$1"
  if [ -n "${base_dir_override}" ]; then
    echo "${base_dir_override}/${task_name}/${variant}"
    return
  fi

  python3 - "$repo_root" "$task_name" "$variant" <<'PY'
from pathlib import Path
import sys
import yaml

repo_root = Path(sys.argv[1])
task_name = sys.argv[2]
variant = sys.argv[3]

task_cfg_path = repo_root / "task_config" / f"{variant}.yml"
with open(task_cfg_path, "r", encoding="utf-8") as f:
    task_cfg = yaml.safe_load(f)

save_path = task_cfg.get("save_path", "./data")
collection_dir = (repo_root / save_path).resolve() / task_name / variant
print(collection_dir)
PY
}

base_dir_args=()
if [ -n "${base_dir_override}" ]; then
  base_dir_args=(--base-dir "${base_dir_override}")
fi

if [ "${dry_run}" -eq 0 ]; then
  mkdir -p "${final_output_dir}"
  mkdir -p "${tmp_output_dir}"
fi

success_count=0
skip_count=0
fail_count=0
dry_count=0

echo "============== Export Actions =============="
echo "Variant/task_config: ${variant}"
echo "Robotwin data      : ${robotwin_data:-<not set>}"
echo "Tasks root         : ${tasks_root}"
echo "Collection root    : ${base_dir_override:-<from task_config save_path>}"
echo "Output dir         : ${final_output_dir}"
echo "Tmp dir            : ${tmp_output_dir}"
echo "Overwrite          : ${overwrite}"
echo "Dry run            : ${dry_run}"
echo "============================================"

for task_name in "${tasks[@]}"; do
  if ! task_id="$(task_id_for_name "${task_name}")"; then
    echo "[SKIP] unknown task name, not in TASK_ORDER: ${task_name}"
    skip_count=$((skip_count + 1))
    echo
    continue
  fi

  collection_dir="$(resolve_collection_dir "${task_name}")"
  data_dir="${collection_dir}/data"

  echo "=========================================="
  echo "Task ${task_id}: ${task_name}"
  echo "Variant/task_config: ${variant}"
  echo "Collection dir: ${collection_dir}"
  echo "=========================================="

  if [ ! -d "${data_dir}" ]; then
    echo "[SKIP] missing data dir: ${data_dir}"
    skip_count=$((skip_count + 1))
    echo
    continue
  fi

  shopt -s nullglob
  episode_files=("${data_dir}"/episode*.hdf5)
  shopt -u nullglob

  if [ ${#episode_files[@]} -eq 0 ]; then
    echo "[SKIP] no episode hdf5 files found"
    skip_count=$((skip_count + 1))
    echo
    continue
  fi

  for hdf5_file in "${episode_files[@]}"; do
    filename="$(basename "${hdf5_file}")"
    episode_num="${filename#episode}"
    episode_num="${episode_num%.hdf5}"
    episode_id="$(printf "%03d" "${episode_num}")"

    dst_json="${final_output_dir}/${task_id}${episode_id}.json"

    if [ -f "${dst_json}" ] && [ "${overwrite}" -eq 0 ]; then
      echo "[SKIP] already exists: ${dst_json}"
      skip_count=$((skip_count + 1))
      echo
      continue
    fi

    echo "Exporting ${task_name} episode ${episode_num} (${variant}) -> ${task_id}${episode_id}.json"

    if [ "${dry_run}" -eq 1 ]; then
      echo "[DRY] python3 ${script_path} ${base_dir_args[*]} --task-name ${task_name} --task-config ${variant} --episode ${episode_num} --output-dir ${tmp_output_dir}"
      echo "[DRY] move ${tmp_output_dir}/episode${episode_num}_actions.json -> ${dst_json}"
      dry_count=$((dry_count + 1))
      echo
      continue
    fi

    if python3 "${script_path}" \
      "${base_dir_args[@]}" \
      --task-name "${task_name}" \
      --task-config "${variant}" \
      --episode "${episode_num}" \
      --output-dir "${tmp_output_dir}"; then

      src_json="${tmp_output_dir}/episode${episode_num}_actions.json"

      if [ -f "${src_json}" ]; then
        mv -f "${src_json}" "${dst_json}"
        success_count=$((success_count + 1))
        echo "[OK] ${dst_json}"
      else
        echo "[FAILED] expected json not found: ${src_json}"
        fail_count=$((fail_count + 1))
      fi
    else
      echo "[FAILED] export failed for ${task_name} episode ${episode_num}"
      fail_count=$((fail_count + 1))
    fi

    echo
  done
done

if [ "${cleanup_tmp}" -eq 1 ] && [ "${dry_run}" -eq 0 ]; then
  rm -rf "${tmp_output_dir}"
fi

echo "============== Summary =============="
echo "Variant/task_config: ${variant}"
echo "Collection root: ${base_dir_override:-<from task_config save_path>}"
echo "Success json files: ${success_count}"
echo "Dry-run planned files: ${dry_count}"
echo "Skipped tasks/episodes: ${skip_count}"
echo "Failed exports: ${fail_count}"
echo "Final output dir: ${final_output_dir}"
echo "====================================="
