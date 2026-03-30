#!/bin/bash
# cd /Users/oooer/ws/RoboTwin/policy/pi0
# bash eval_collect_parallel.sh \
#   grab_roller \
#   wm_agilex_100 \
#   pi0_base_aloha_robotwin_full \
#   pi0_grab_roller \
#   4 \
#   50 \
#   100000 \
#   0
# 4: 用 4 张 GPU 并行
# 50: 总共收 50 个 episode，不是每张卡 50 个
# 100000: 起始 seed
# 0: GPU 起始编号，表示用 0,1,2,3

set -euo pipefail

task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
num_gpus=${5:-4}
episode_num=${6:-50}
base_seed=${7:-100000}
gpu_offset=${8:-0}
extra_args=("${@:9}")

declare -A gpu_pid

for (( worker=0; worker<num_gpus; worker++ )); do
  gpu=$((gpu_offset + worker))
  seed_start=$((base_seed + worker))
  echo "[$(date +%T)] 启动 GPU $gpu -> ${task_name} (seed_start=${seed_start}, stride=${num_gpus}, target=${episode_num})"
  bash eval.sh \
    "${task_name}" \
    "${task_config}" \
    "${train_config_name}" \
    "${model_name}" \
    0 \
    "${gpu}" \
    --episode_num "${episode_num}" \
    --seed_start "${seed_start}" \
    --seed_stride "${num_gpus}" \
    "${extra_args[@]}" &
  gpu_pid[$gpu]=$!
done

status=0
for gpu in "${!gpu_pid[@]}"; do
  pid=${gpu_pid[$gpu]}
  if ! wait "${pid}"; then
    echo "[$(date +%T)] GPU $gpu 任务失败 (pid=${pid})"
    status=1
  fi
done

if [[ ${status} -eq 0 ]]; then
  echo "✅ ${task_name} 并行 eval 收集完成，总目标 episode 数: ${episode_num}"
else
  echo "❌ ${task_name} 并行 eval 收集有失败进程"
fi

exit ${status}
