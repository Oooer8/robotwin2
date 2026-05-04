#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G

policy_name=pi0
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}

is_integer() {
    [[ "$1" =~ ^-?[0-9]+$ ]]
}

if [[ $# -ge 7 ]] && is_integer "${5}" && is_integer "${6}" && is_integer "${7}"; then
    checkpoint_id=${5}
    seed=${6}
    gpu_id=${7}
    extra_args=("${@:8}")
else
    checkpoint_id=${CHECKPOINT_ID:-30000}
    seed=${5}
    gpu_id=${6}
    extra_args=("${@:7}")
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mcheckpoint id: ${checkpoint_id}\033[0m"

source .venv/bin/activate
cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --checkpoint_id ${checkpoint_id} \
    --ckpt_setting ${model_name}_${checkpoint_id} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    "${extra_args[@]}"
