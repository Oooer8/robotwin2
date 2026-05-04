# CosmosPolicyRemote

This policy directory was generated from the `cosmos-policy` repo to let RoboTwin / RobotWin
query a remote Cosmos Policy server over HTTP.

## Files

- `deploy_policy.py`: RobotWin policy adapter
- `__init__.py`: package-style exports for RobotWin loaders
- `CosmosPolicyRemote.py`: compatibility shim when the policy directory itself is on `PYTHONPATH`
- `deploy_policy.yml`: adapter configuration
- `eval.sh`: convenience wrapper around `python script/eval_policy.py`
- `CosmosPolicyRemote.py` in the RoboTwin repo root: compatibility shim when the repo root is on `PYTHONPATH`

## Expected Server

Start the server from the `cosmos-policy` repo with a RobotWin Agilex checkpoint and point it at:

`http://127.0.0.1:8777/act`

## Typical Usage

```bash
cd /root/workspace/RoboTwin
bash policy/CosmosPolicyRemote/eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id>
```

If RobotWin camera names or proprio fields differ from the defaults, edit `deploy_policy.yml`.
