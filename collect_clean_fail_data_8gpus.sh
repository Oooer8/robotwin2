#!/bin/bash
# ============================================================
# 采集 50 个任务、每个任务 50 条 "clean 失败" 数据。
#
# 原理: 失败数据不是从零采集, 而是【重放已采集的 clean 成功轨迹 + 加噪声】,
#       只保留 "规划成功但执行失败" (plan_success and not exec_success) 的 episode。
#       详见 script/collect_fail_data.py。
#
# 前置条件 (必须先完成):
#   先用 clean 配置采集成功数据 (例如 bash collect_clean_data_8gpus.sh 8),
#   使得每个任务存在以下源数据:
#     data/<task_name>/demo_clean/_traj_data/episode*.pkl
#     data/<task_name>/demo_clean/seed.txt
#
# 本脚本会把这些成功轨迹作为 source, 采集对应的 clean 失败数据。
#
# 使用 task_config/demo_clean.yml 保证场景为 clean (无域随机化):
#   - episode_num: 50           -> 每个任务最多采 50 条失败
#   - domain_randomization 全关  -> clean 场景
#   - data_type.rgb: true       -> 自动导出 head/left/right 相机视频 (mp4)
#   - data_type.endpose: true   -> 保存 eef (末端位姿) 到 hdf5
#   - data_type.qpos: true      -> 保存真实关节到 /joint_action, 目标关节到 /joint_target
#   - collect_data: true        -> 自动生成 hdf5 + video
#
# 产物目录 (save_path 默认 ./data):
#   data/<task_name>/demo_clean_fail/data/episode{i}.hdf5              (含 eef/endpose)
#   data/<task_name>/demo_clean_fail/video/episode{i}.mp4             (head 相机)
#   data/<task_name>/demo_clean_fail/video/episode{i}_left_camera.mp4
#   data/<task_name>/demo_clean_fail/video/episode{i}_right_camera.mp4
#   data/<task_name>/demo_clean_fail/seed.txt / scene_info.json / instructions/
#
# 用法:
#   bash collect_clean_fail_data_8gpus.sh [num_gpus] [source_collection_suffix] [save_path] [extra fail args...]
#   # 例: bash collect_clean_fail_data_8gpus.sh 8
#   # 例: bash collect_clean_fail_data_8gpus.sh 8 demo_clean /abs/path/to/data
#
# 说明:
#   - task_config 固定为 demo_clean (clean 场景)。
#   - source_collection_suffix 默认 demo_clean, 即上一步 clean 成功数据所在目录名。
#   - 目标目录默认 <source_collection_suffix>_fail (即 demo_clean_fail)。
#   - 底层调用 collect_fail_data_agilex_8gpus.sh, 它会自动发现所有已存在源数据的任务。
# ============================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

task_config="demo_clean"
num_gpus="${1:-8}"
source_collection_suffix="${2:-demo_clean}"
save_path_override="${3:-}"
extra_fail_args=("${@:4}")

echo "[INFO] task_config=$task_config (clean 场景)"
echo "[INFO] source_collection_suffix=$source_collection_suffix"
echo "[INFO] num_gpus=$num_gpus"
echo "[INFO] target=default ${source_collection_suffix}_fail"

bash collect_fail_data_agilex_8gpus.sh \
  "$task_config" \
  "$num_gpus" \
  "$source_collection_suffix" \
  "$save_path_override" \
  "${extra_fail_args[@]}"
