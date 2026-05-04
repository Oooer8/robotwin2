#!/bin/bash
set -euo pipefail

policy_name="CosmosPolicyRemote"
task_name="${1:?task_name is required}"
task_config="${2:?task_config is required}"
ckpt_setting="${3:?ckpt_setting is required}"
seed="${4:?seed is required}"
gpu_id="${5:?gpu_id is required}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/..:${SCRIPT_DIR}/../..${PYTHONPATH:+:${PYTHONPATH}}"

python - "${policy_name}" <<'PY'
import importlib
import sys

policy_name = sys.argv[1]
policy_module = importlib.import_module(policy_name)
required_symbols = ("get_model", "eval", "update_obs")
missing_symbols = [name for name in required_symbols if not hasattr(policy_module, name)]
if missing_symbols:
    module_path = getattr(policy_module, "__file__", "<namespace package>")
    raise SystemExit(
        f"Policy module {policy_name} loaded from {module_path} is missing required exports: {missing_symbols}"
    )
print(f"Loaded policy module {policy_name} from {getattr(policy_module, '__file__', '<namespace package>')}")
PY

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py \
  --config "policy/${policy_name}/deploy_policy.yml" \
  --overrides \
  --task_name "${task_name}" \
  --task_config "${task_config}" \
  --ckpt_setting "${ckpt_setting}" \
  --seed "${seed}" \
  --policy_name "${policy_name}"
