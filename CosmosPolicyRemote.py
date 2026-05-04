"""Compatibility shim for loaders that import ``CosmosPolicyRemote`` from the RoboTwin repo root."""

from __future__ import annotations

import sys
from pathlib import Path

_POLICY_DIR = Path(__file__).resolve().parent / "policy" / "CosmosPolicyRemote"
if str(_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(_POLICY_DIR))

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
