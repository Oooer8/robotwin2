#!/bin/bash
# 使用方法
# 1. 可选传入 task_config，例如:
#    bash export_all_action.sh wm_agilex_100_random
# 2. 可选再传 base_dir 覆盖 yml 里的 save_path，例如:
#    bash export_all_action.sh wm_agilex_100_fail /root/workspace/robotwin_data/agilex_eval
# 3. 或者直接修改下面的默认值
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "${script_dir}/task_config" ]; then
  repo_root="${script_dir}"
elif [ -d "${script_dir}/../task_config" ]; then
  repo_root="$(cd "${script_dir}/.." && pwd)"
else
  echo "Cannot locate repo root from ${script_dir}" >&2
  exit 1
fi

default_task_config="wm_agilex_100"
task_config="${1:-${default_task_config}}"
base_dir_override="${2:-}"
base_dir_args=()
if [ -n "${base_dir_override}" ]; then
  base_dir_args=(--base-dir "${base_dir_override}")
fi

if [ -f "${script_dir}/export_actions_json.py" ]; then
  script_path="${script_dir}/export_actions_json.py"
else
  echo "Cannot locate export_actions_json.py in repo root" >&2
  exit 1
fi

final_output_dir="/root/workspace/action"
tmp_output_dir="/root/workspace/robotwin_data/action_tmp"

mkdir -p "${final_output_dir}"
mkdir -p "${tmp_output_dir}"

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

success_count=0
skip_count=0
fail_count=0

resolve_collection_dir() {
  local task_name="$1"
  if [ -n "${base_dir_override}" ]; then
    echo "${base_dir_override}/${task_name}/${task_config}"
    return
  fi

  python3 - "$repo_root" "$task_name" "$task_config" <<'PY'
from pathlib import Path
import sys
import yaml

repo_root = Path(sys.argv[1])
task_name = sys.argv[2]
task_config = sys.argv[3]

task_cfg_path = repo_root / "task_config" / f"{task_config}.yml"
with open(task_cfg_path, "r", encoding="utf-8") as f:
    task_cfg = yaml.safe_load(f)

save_path = task_cfg.get("save_path", "./data")
collection_dir = (repo_root / save_path).resolve() / task_name / task_config
print(collection_dir)
PY
}

for ((i=0; i<${#tasks[@]}; i++)); do
  task_name="${tasks[i]}"
  task_id=$(printf "%03d" $((i + 1)))
  collection_dir="$(resolve_collection_dir "${task_name}")"
  data_dir="${collection_dir}/data"

  echo "=========================================="
  echo "Task ${task_id}: ${task_name}"
  echo "Task config: ${task_config}"
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
    filename=$(basename "${hdf5_file}")
    episode_num=${filename#episode}
    episode_num=${episode_num%.hdf5}
    episode_id=$(printf "%03d" "${episode_num}")

    dst_json="${final_output_dir}/${task_id}${episode_id}.json"

    # ✅ 目标 JSON 已存在则跳过，不重复处理
    if [ -f "${dst_json}" ]; then
      echo "[SKIP] already exists: ${dst_json}"
      skip_count=$((skip_count + 1))
      echo
      continue
    fi

    echo "Exporting ${task_name} episode ${episode_num} (${task_config}) -> ${task_id}${episode_id}.json"

    if python3 "${script_path}" \
      "${base_dir_args[@]}" \
      --task-name "${task_name}" \
      --task-config "${task_config}" \
      --episode "${episode_num}" \
      --output-dir "${tmp_output_dir}"; then

      src_json="${tmp_output_dir}/episode${episode_num}_actions.json"

      if [ -f "${src_json}" ]; then
        mv "${src_json}" "${dst_json}"
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

rm -rf "${tmp_output_dir}"

echo "============== Summary =============="
echo "Task config: ${task_config}"
echo "Base dir override: ${base_dir_override:-<from task_config save_path>}"
echo "Success json files: ${success_count}"
echo "Skipped tasks/episodes: ${skip_count}"
echo "Failed exports: ${fail_count}"
echo "Final output dir: ${final_output_dir}"
echo "====================================="