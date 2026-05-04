"""Compatibility shim for loaders that add this policy directory to ``PYTHONPATH``."""

from deploy_policy import (
    RemoteRobotWinPolicy,
    encode_obs,
    eval,
    get_action,
    get_model,
    reset_model,
    update_obs,
)

__all__ = ["RemoteRobotWinPolicy", "encode_obs", "eval", "get_action", "get_model", "reset_model", "update_obs"]
