import sys

sys.path.append("./")

import importlib
import json
import os
import time
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from types import MethodType

import numpy as np
import transforms3d as t3d
import yaml

from envs import *
from envs.utils.pkl2hdf5 import process_folder_to_hdf5_video
from recover_episode_instructions import recover_instructions


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def get_random_action_args(args):
    random_action_args = deepcopy(args.get("random_action", {}))
    random_action_args.setdefault("action_type", "ee")
    random_action_args.setdefault("joint_delta_low", -0.25)
    random_action_args.setdefault("joint_delta_high", 0.25)
    random_action_args.setdefault("joint_clip", np.pi)
    random_action_args.setdefault("ee_delta_low", [-0.06, -0.05, -0.05])
    random_action_args.setdefault("ee_delta_high", [0.06, 0.05, 0.05])
    random_action_args.setdefault("head_visible_workspace", True)
    random_action_args.setdefault("left_xyz_min", [-0.28, -0.18, 0.82])
    random_action_args.setdefault("left_xyz_max", [0.02, 0.18, 1.08])
    random_action_args.setdefault("right_xyz_min", [-0.02, -0.18, 0.82])
    random_action_args.setdefault("right_xyz_max", [0.28, 0.18, 1.08])
    random_action_args.setdefault("center_bias_prob", 0.8)
    random_action_args.setdefault("center_box_scale", 0.55)
    random_action_args.setdefault("random_orientation", True)
    random_action_args.setdefault("orientation_mode", "delta")
    random_action_args.setdefault("delta_roll_range", random_action_args.get("roll_range", [-0.35, 0.35]))
    random_action_args.setdefault("delta_pitch_range", random_action_args.get("pitch_range", [-0.45, 0.45]))
    random_action_args.setdefault("delta_yaw_range", random_action_args.get("yaw_range", [-1.2, 1.2]))
    random_action_args.setdefault("gripper_hold_prob", 0.7)
    random_action_args.setdefault("gripper_open_prob", 0.15)
    random_action_args.setdefault("gripper_close_prob", 0.15)
    random_action_args.setdefault("frame_limit", random_action_args.get("step_limit", 300))
    return random_action_args


def get_episode_frame_limit(random_action_args):
    frame_limit = random_action_args.get("frame_limit")
    if frame_limit is None:
        return None
    return int(frame_limit)


def sample_random_gripper_target(current_val, random_action_args):
    hold_prob = random_action_args["gripper_hold_prob"]
    open_prob = random_action_args["gripper_open_prob"]
    close_prob = random_action_args["gripper_close_prob"]
    total_prob = hold_prob + open_prob + close_prob

    if total_prob <= 0:
        return float(np.clip(current_val, 0.0, 1.0))

    prob = np.array([hold_prob, open_prob, close_prob], dtype=np.float64) / total_prob
    choice = np.random.choice(["hold", "open", "close"], p=prob)
    if choice == "hold":
        return float(np.clip(current_val, 0.0, 1.0))
    if choice == "open":
        return 1.0
    return 0.0


def sample_random_qpos_action(task_env, random_action_args):
    left_jointstate = np.asarray(task_env.robot.get_left_arm_jointState(), dtype=np.float32)
    right_jointstate = np.asarray(task_env.robot.get_right_arm_jointState(), dtype=np.float32)

    left_arm_current = left_jointstate[:-1]
    right_arm_current = right_jointstate[:-1]
    left_gripper_current = float(left_jointstate[-1])
    right_gripper_current = float(right_jointstate[-1])

    joint_delta_low = float(random_action_args["joint_delta_low"])
    joint_delta_high = float(random_action_args["joint_delta_high"])
    joint_clip = float(random_action_args["joint_clip"])

    left_arm_target = left_arm_current + np.random.uniform(
        low=joint_delta_low,
        high=joint_delta_high,
        size=left_arm_current.shape,
    )
    right_arm_target = right_arm_current + np.random.uniform(
        low=joint_delta_low,
        high=joint_delta_high,
        size=right_arm_current.shape,
    )

    left_arm_target = np.clip(left_arm_target, -joint_clip, joint_clip)
    right_arm_target = np.clip(right_arm_target, -joint_clip, joint_clip)

    left_gripper_target = sample_random_gripper_target(left_gripper_current, random_action_args)
    right_gripper_target = sample_random_gripper_target(right_gripper_current, random_action_args)

    return np.concatenate(
        [
            left_arm_target,
            np.array([left_gripper_target], dtype=np.float32),
            right_arm_target,
            np.array([right_gripper_target], dtype=np.float32),
        ]
    ).astype(np.float32)


def sample_visible_xyz(current_pose, xyz_min, xyz_max, delta_low, delta_high, center_bias_prob, center_box_scale):
    target_xyz = current_pose[:3] + np.random.uniform(low=delta_low, high=delta_high, size=3)
    target_xyz = np.clip(target_xyz, xyz_min, xyz_max)

    if np.random.rand() < center_bias_prob:
        center = (xyz_min + xyz_max) / 2
        half_range = (xyz_max - xyz_min) * center_box_scale / 2
        inner_min = center - half_range
        inner_max = center + half_range
        target_xyz = np.clip(target_xyz, inner_min, inner_max)

    return target_xyz.astype(np.float32)


def sample_random_quaternion(current_quat, random_action_args):
    if not bool(random_action_args["random_orientation"]):
        return np.asarray(current_quat, dtype=np.float32)

    orientation_mode = random_action_args.get("orientation_mode", "delta")
    if orientation_mode != "delta":
        raise ValueError(f"Unsupported orientation_mode: {orientation_mode}")

    delta_roll = np.random.uniform(
        low=float(random_action_args["delta_roll_range"][0]),
        high=float(random_action_args["delta_roll_range"][1]),
    )
    delta_pitch = np.random.uniform(
        low=float(random_action_args["delta_pitch_range"][0]),
        high=float(random_action_args["delta_pitch_range"][1]),
    )
    delta_yaw = np.random.uniform(
        low=float(random_action_args["delta_yaw_range"][0]),
        high=float(random_action_args["delta_yaw_range"][1]),
    )

    base_rot = t3d.quaternions.quat2mat(np.asarray(current_quat, dtype=np.float64))
    delta_rot = t3d.euler.euler2mat(delta_roll, delta_pitch, delta_yaw)
    target_rot = base_rot @ delta_rot
    return np.asarray(t3d.quaternions.mat2quat(target_rot), dtype=np.float32)


def sample_random_ee_action(task_env, random_action_args):
    left_pose = np.asarray(task_env.robot.get_left_ee_pose(), dtype=np.float32)
    right_pose = np.asarray(task_env.robot.get_right_ee_pose(), dtype=np.float32)

    delta_low = np.asarray(random_action_args["ee_delta_low"], dtype=np.float32)
    delta_high = np.asarray(random_action_args["ee_delta_high"], dtype=np.float32)
    center_bias_prob = float(random_action_args["center_bias_prob"])
    center_box_scale = float(random_action_args["center_box_scale"])

    if bool(random_action_args["head_visible_workspace"]):
        left_xyz = sample_visible_xyz(
            left_pose,
            np.asarray(random_action_args["left_xyz_min"], dtype=np.float32),
            np.asarray(random_action_args["left_xyz_max"], dtype=np.float32),
            delta_low,
            delta_high,
            center_bias_prob,
            center_box_scale,
        )
        right_xyz = sample_visible_xyz(
            right_pose,
            np.asarray(random_action_args["right_xyz_min"], dtype=np.float32),
            np.asarray(random_action_args["right_xyz_max"], dtype=np.float32),
            delta_low,
            delta_high,
            center_bias_prob,
            center_box_scale,
        )
    else:
        left_xyz = left_pose[:3] + np.random.uniform(low=delta_low, high=delta_high, size=3)
        right_xyz = right_pose[:3] + np.random.uniform(low=delta_low, high=delta_high, size=3)

    left_quat = sample_random_quaternion(left_pose[3:], random_action_args)
    right_quat = sample_random_quaternion(right_pose[3:], random_action_args)

    left_gripper_target = sample_random_gripper_target(task_env.robot.get_left_gripper_val(), random_action_args)
    right_gripper_target = sample_random_gripper_target(task_env.robot.get_right_gripper_val(), random_action_args)

    return np.concatenate(
        [
            left_xyz,
            left_quat,
            np.array([left_gripper_target], dtype=np.float32),
            right_xyz,
            right_quat,
            np.array([right_gripper_target], dtype=np.float32),
        ]
    ).astype(np.float32)


def take_picture_with_limit(task_env, frame_limit):
    if frame_limit is not None and task_env.FRAME_IDX >= frame_limit:
        return False
    task_env._take_picture()
    return True


def take_action_with_save(task_env, action, frame_limit, action_type="qpos"):
    if task_env.eval_success:
        return

    if frame_limit is not None and task_env.FRAME_IDX >= frame_limit:
        return

    eval_video_freq = 1
    save_freq = task_env.save_freq
    if task_env.eval_video_path is not None and task_env.take_action_cnt % eval_video_freq == 0:
        task_env.eval_video_ffmpeg.stdin.write(task_env.now_obs["observation"]["head_camera"]["rgb"].tobytes())

    task_env.take_action_cnt += 1
    frame_msg = "inf" if frame_limit is None else str(frame_limit)
    print(
        f"action: \033[92m{task_env.take_action_cnt}\033[0m | "
        f"frame: \033[92m{task_env.FRAME_IDX} / {frame_msg}\033[0m",
        end="\r",
    )

    task_env._update_render()
    if task_env.render_freq:
        task_env.viewer.render()

    actions = np.array([action])
    left_jointstate = task_env.robot.get_left_arm_jointState()
    right_jointstate = task_env.robot.get_right_arm_jointState()
    left_arm_dim = len(left_jointstate) - 1 if action_type == "qpos" else 7
    right_arm_dim = len(right_jointstate) - 1 if action_type == "qpos" else 7
    current_jointstate = np.array(left_jointstate + right_jointstate)

    left_arm_actions, left_gripper_actions, left_current_qpos, left_path = ([], [], [], [])
    right_arm_actions, right_gripper_actions, right_current_qpos, right_path = ([], [], [], [])

    left_arm_actions, left_gripper_actions = (
        actions[:, :left_arm_dim],
        actions[:, left_arm_dim],
    )
    right_arm_actions, right_gripper_actions = (
        actions[:, left_arm_dim + 1:left_arm_dim + right_arm_dim + 1],
        actions[:, left_arm_dim + right_arm_dim + 1],
    )
    left_current_gripper, right_current_gripper = (
        task_env.robot.get_left_gripper_val(),
        task_env.robot.get_right_gripper_val(),
    )

    left_gripper_path = np.hstack((left_current_gripper, left_gripper_actions))
    right_gripper_path = np.hstack((right_current_gripper, right_gripper_actions))

    if action_type == "qpos":
        left_current_qpos, right_current_qpos = (
            current_jointstate[:left_arm_dim],
            current_jointstate[left_arm_dim + 1:left_arm_dim + right_arm_dim + 1],
        )
        left_path = np.vstack((left_current_qpos, left_arm_actions))
        right_path = np.vstack((right_current_qpos, right_arm_actions))

        topp_left_flag, topp_right_flag = True, True

        try:
            times, left_pos, left_vel, acc, duration = task_env.robot.left_mplib_planner.TOPP(
                left_path,
                1 / 250,
                verbose=True,
            )
            left_result = {"position": left_pos, "velocity": left_vel}
            left_n_step = left_result["position"].shape[0]
        except Exception:
            topp_left_flag = False
            left_n_step = 50

        if left_n_step == 0:
            topp_left_flag = False
            left_n_step = 50

        try:
            times, right_pos, right_vel, acc, duration = task_env.robot.right_mplib_planner.TOPP(
                right_path,
                1 / 250,
                verbose=True,
            )
            right_result = {"position": right_pos, "velocity": right_vel}
            right_n_step = right_result["position"].shape[0]
        except Exception:
            topp_right_flag = False
            right_n_step = 50

        if right_n_step == 0:
            topp_right_flag = False
            right_n_step = 50

    elif action_type == "ee":
        left_result = task_env.robot.left_plan_path(left_arm_actions[0])
        right_result = task_env.robot.right_plan_path(right_arm_actions[0])
        if left_result["status"] != "Success":
            left_n_step = 50
            topp_left_flag = False
        else:
            left_n_step = left_result["position"].shape[0]
            topp_left_flag = True

        if right_result["status"] != "Success":
            right_n_step = 50
            topp_right_flag = False
        else:
            right_n_step = right_result["position"].shape[0]
            topp_right_flag = True
    else:
        raise ValueError(f"Unsupported action_type: {action_type}")

    left_mod_num = left_n_step % len(left_gripper_actions)
    right_mod_num = right_n_step % len(right_gripper_actions)
    left_gripper_step = [0] + [
        left_n_step // len(left_gripper_actions) + (1 if i < left_mod_num else 0)
        for i in range(len(left_gripper_actions))
    ]
    right_gripper_step = [0] + [
        right_n_step // len(right_gripper_actions) + (1 if i < right_mod_num else 0)
        for i in range(len(right_gripper_actions))
    ]

    left_gripper = []
    for gripper_step in range(1, left_gripper_path.shape[0]):
        region_left_gripper = np.linspace(
            left_gripper_path[gripper_step - 1],
            left_gripper_path[gripper_step],
            left_gripper_step[gripper_step] + 1,
        )[1:]
        left_gripper = left_gripper + region_left_gripper.tolist()
    left_gripper = np.array(left_gripper)

    right_gripper = []
    for gripper_step in range(1, right_gripper_path.shape[0]):
        region_right_gripper = np.linspace(
            right_gripper_path[gripper_step - 1],
            right_gripper_path[gripper_step],
            right_gripper_step[gripper_step] + 1,
        )[1:]
        right_gripper = right_gripper + region_right_gripper.tolist()
    right_gripper = np.array(right_gripper)

    now_left_id, now_right_id = 0, 0
    control_idx = 0

    if save_freq is not None:
        if not take_picture_with_limit(task_env, frame_limit):
            return

    while now_left_id < left_n_step or now_right_id < right_n_step:
        if now_left_id < left_n_step and now_left_id / left_n_step <= now_right_id / right_n_step:
            if topp_left_flag:
                task_env.robot.set_arm_joints(
                    left_result["position"][now_left_id],
                    left_result["velocity"][now_left_id],
                    "left",
                )
            task_env.robot.set_gripper(left_gripper[now_left_id], "left")
            now_left_id += 1

        if now_right_id < right_n_step and now_right_id / right_n_step <= now_left_id / left_n_step:
            if topp_right_flag:
                task_env.robot.set_arm_joints(
                    right_result["position"][now_right_id],
                    right_result["velocity"][now_right_id],
                    "right",
                )
            task_env.robot.set_gripper(right_gripper[now_right_id], "right")
            now_right_id += 1

        task_env.scene.step()
        task_env._update_render()

        if save_freq is not None and control_idx % save_freq == 0:
            if not take_picture_with_limit(task_env, frame_limit):
                return
        control_idx += 1

        if task_env.check_success():
            task_env.eval_success = True
            task_env.get_obs()
            if save_freq is not None:
                take_picture_with_limit(task_env, frame_limit)
            if task_env.eval_video_path is not None:
                task_env.eval_video_ffmpeg.stdin.write(task_env.now_obs["observation"]["head_camera"]["rgb"].tobytes())
            return

    task_env._update_render()
    if save_freq is not None:
        take_picture_with_limit(task_env, frame_limit)
    if task_env.render_freq:
        task_env.viewer.render()


def ensure_scene_info_file(save_path):
    info_file_path = os.path.join(save_path, "scene_info.json")
    if not os.path.exists(info_file_path):
        with open(info_file_path, "w", encoding="utf-8") as file:
            json.dump({}, file, ensure_ascii=False)
    return info_file_path


def load_seed_list(save_path):
    seed_path = os.path.join(save_path, "seed.txt")
    if not os.path.exists(seed_path):
        return []
    with open(seed_path, "r", encoding="utf-8") as file:
        seed_list = file.read().split()
    return [int(i) for i in seed_list]


def save_seed_list(save_path, seed_list):
    seed_path = os.path.join(save_path, "seed.txt")
    with open(seed_path, "w", encoding="utf-8") as file:
        for seed in seed_list:
            file.write(f"{seed} ")


def exist_hdf5(save_path, idx):
    file_path = os.path.join(save_path, "data", f"episode{idx}.hdf5")
    return os.path.exists(file_path)


def merge_random_episode(task_env, save_path, episode_idx):
    cache_path = task_env.folder_path["cache"]
    target_file_path = os.path.join(save_path, "data", f"episode{episode_idx}.hdf5")
    target_video_path = os.path.join(save_path, "video", f"episode{episode_idx}.mp4")
    os.makedirs(os.path.join(save_path, "data"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "video"), exist_ok=True)
    process_folder_to_hdf5_video(cache_path, target_file_path, target_video_path)


def collect_random_episode(task_env, task_name, task_config, seed, random_action_args):
    frame_limit = get_episode_frame_limit(random_action_args)
    if task_env.save_freq is None:
        raise ValueError("save_freq cannot be None when collecting random data with frame_limit.")
    action_type = random_action_args["action_type"]

    while not task_env.eval_success and (frame_limit is None or task_env.FRAME_IDX < frame_limit):
        if action_type == "qpos":
            random_action = sample_random_qpos_action(task_env, random_action_args)
        elif action_type == "ee":
            random_action = sample_random_ee_action(task_env, random_action_args)
        else:
            raise ValueError(f"Unsupported random action_type: {action_type}")
        take_action_with_save(task_env, random_action, frame_limit=frame_limit, action_type=action_type)

    info = deepcopy(task_env.info)
    info["collector"] = "random_actions"
    info["task_name"] = task_name
    info["task_config"] = task_config
    info["seed"] = seed
    info["frame_limit"] = frame_limit
    info["saved_frame_count"] = int(task_env.FRAME_IDX)
    info["sampled_action_count"] = int(task_env.take_action_cnt)
    info["success"] = bool(task_env.check_success())
    info["plan_success"] = None
    info["random_action"] = {
        "action_type": action_type,
        "joint_delta_low": float(random_action_args["joint_delta_low"]),
        "joint_delta_high": float(random_action_args["joint_delta_high"]),
        "joint_clip": float(random_action_args["joint_clip"]),
        "ee_delta_low": [float(v) for v in random_action_args["ee_delta_low"]],
        "ee_delta_high": [float(v) for v in random_action_args["ee_delta_high"]],
        "head_visible_workspace": bool(random_action_args["head_visible_workspace"]),
        "left_xyz_min": [float(v) for v in random_action_args["left_xyz_min"]],
        "left_xyz_max": [float(v) for v in random_action_args["left_xyz_max"]],
        "right_xyz_min": [float(v) for v in random_action_args["right_xyz_min"]],
        "right_xyz_max": [float(v) for v in random_action_args["right_xyz_max"]],
        "center_bias_prob": float(random_action_args["center_bias_prob"]),
        "center_box_scale": float(random_action_args["center_box_scale"]),
        "random_orientation": bool(random_action_args["random_orientation"]),
        "orientation_mode": str(random_action_args["orientation_mode"]),
        "delta_roll_range": [float(v) for v in random_action_args["delta_roll_range"]],
        "delta_pitch_range": [float(v) for v in random_action_args["delta_pitch_range"]],
        "delta_yaw_range": [float(v) for v in random_action_args["delta_yaw_range"]],
        "gripper_hold_prob": float(random_action_args["gripper_hold_prob"]),
        "gripper_open_prob": float(random_action_args["gripper_open_prob"]),
        "gripper_close_prob": float(random_action_args["gripper_close_prob"]),
    }
    return info


def pack_joint_state(left_jointstate, right_jointstate):
    left_jointstate = np.asarray(left_jointstate, dtype=np.float32)
    right_jointstate = np.asarray(right_jointstate, dtype=np.float32)
    return {
        "left_arm": left_jointstate[:-1],
        "left_gripper": np.float32(left_jointstate[-1]),
        "right_arm": right_jointstate[:-1],
        "right_gripper": np.float32(right_jointstate[-1]),
        "vector": np.concatenate([left_jointstate, right_jointstate]).astype(np.float32),
    }


def patch_get_obs_with_real_joint_state(task_env):
    if not hasattr(task_env, "_original_get_obs_for_random_real_state"):
        task_env._original_get_obs_for_random_real_state = task_env.get_obs

    original_get_obs = task_env._original_get_obs_for_random_real_state

    def get_obs_with_real_joint_state(self):
        obs = original_get_obs()
        if not self.data_type.get("qpos", False):
            return obs

        left_target = self.robot.get_left_arm_jointState()
        right_target = self.robot.get_right_arm_jointState()
        left_real = self.robot.get_left_arm_real_jointState()
        right_real = self.robot.get_right_arm_real_jointState()

        obs["joint_action"] = pack_joint_state(left_real, right_real)
        obs["joint_target"] = pack_joint_state(left_target, right_target)
        self.now_obs = deepcopy(obs)
        return obs

    task_env.get_obs = MethodType(get_obs_with_real_joint_state, task_env)


def build_args(task_name, task_config):
    config_path = f"./task_config/{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_name):
        robot_file = embodiment_types[embodiment_name]["file_path"]
        if robot_file is None:
            raise RuntimeError("missing embodiment files")
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("number of embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    args["embodiment_name"] = embodiment_name
    args["task_config"] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    return args, embodiment_name


def print_config(args, embodiment_name, random_action_args):
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))
    print(
        "\033[94mHead Camera Config:\033[0m "
        + str(args["camera"]["head_camera_type"])
        + f", {args['camera']['collect_head_camera']}"
    )
    print(
        "\033[94mWrist Camera Config:\033[0m "
        + str(args["camera"]["wrist_camera_type"])
        + f", {args['camera']['collect_wrist_camera']}"
    )
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\033[94mCollector:\033[0m random_actions_real_state")
    print("\033[94mRandom Action Type:\033[0m " + str(random_action_args["action_type"]))
    print("\033[94mHead-Visible Workspace:\033[0m " + str(bool(random_action_args["head_visible_workspace"])))
    print("\033[94mRandom Frame Limit:\033[0m " + str(get_episode_frame_limit(random_action_args)))
    print("\033[94mJoint Storage:\033[0m real_state -> /joint_action, target -> /joint_target")
    print("\n==================================")


def run(task_env, args, random_action_args):
    os.makedirs(args["save_path"], exist_ok=True)
    seed_list = load_seed_list(args["save_path"])
    fail_num = 0

    if args["use_seed"] and not seed_list:
        raise FileNotFoundError(f"seed.txt not found in {args['save_path']}")

    st_idx = 0
    while exist_hdf5(args["save_path"], st_idx):
        st_idx += 1

    if seed_list:
        print(f"Exist seed file, Start from episode: {st_idx}, recorded seeds: {len(seed_list)}")

    info_file_path = ensure_scene_info_file(args["save_path"])

    def collect_one_episode(episode_idx, seed):
        nonlocal fail_num
        try:
            print(f"\033[34mTask name: {args['task_name']}\033[0m")
            print(f"random data episode {episode_idx} collecting... (seed = {seed})")
            episode_args = deepcopy(args)
            episode_args["need_plan"] = False
            episode_args["save_data"] = True
            episode_args["render_freq"] = 0
            task_env.setup_demo(now_ep_num=episode_idx, seed=seed, **episode_args)
            patch_get_obs_with_real_joint_state(task_env)
            task_env.get_obs()

            info = collect_random_episode(task_env, args["task_name"], args["task_config"], seed, random_action_args)
            info["collector"] = "random_actions_real_state"
            info["joint_storage_mode"] = {
                "joint_action": "real_joint_state",
                "joint_target": "drive_target",
            }

            with open(info_file_path, "r", encoding="utf-8") as file:
                info_db = json.load(file)
            info_db[f"episode_{episode_idx}"] = info
            with open(info_file_path, "w", encoding="utf-8") as file:
                json.dump(info_db, file, ensure_ascii=False, indent=4)

            task_env.close_env(clear_cache=((episode_idx + 1) % args["clear_cache_freq"] == 0))
            merge_random_episode(task_env, args["save_path"], episode_idx)
            task_env.remove_data_cache()
            print(
                f"random data episode {episode_idx} done! "
                f"(seed = {seed}, success = {info['success']}, actions = {info['sampled_action_count']})"
            )
            return True
        except UnStableError as e:
            print(" -------------")
            print(f"random data episode {episode_idx} fail! (seed = {seed})")
            print("Error: ", e)
            print(" -------------")
            fail_num += 1
            task_env.close_env()
            time.sleep(0.3)
        except Exception as e:
            print(" -------------")
            print(f"random data episode {episode_idx} fail! (seed = {seed})")
            print("Error: ", e)
            print(" -------------")
            fail_num += 1
            task_env.close_env()
            time.sleep(1)
        return False

    episode_idx = st_idx

    while episode_idx < min(len(seed_list), args["episode_num"]):
        if collect_one_episode(episode_idx, seed_list[episode_idx]):
            episode_idx += 1
        else:
            break

    if args["use_seed"]:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        print(f"\nComplete random data collection, failed \033[91m{fail_num}\033[0m times \n")
        return

    epid = max(seed_list) + 1 if seed_list else 0

    while episode_idx < args["episode_num"]:
        if collect_one_episode(episode_idx, epid):
            seed_list.append(epid)
            save_seed_list(args["save_path"], seed_list)
            episode_idx += 1
        epid += 1

    try:
        instruction_result = recover_instructions(
            collection_dir=Path(args["save_path"]),
            task_name=args["task_name"],
            task_config=args["task_config"],
            max_num=int(args.get("language_num", 100)),
        )
        print(
            f"Recovered instructions for \033[92m{instruction_result['episodes']}\033[0m episodes at "
            f"\033[92m{instruction_result['instructions_dir']}\033[0m"
        )
    except Exception as e:
        print(f"\033[93mWarning:\033[0m failed to generate instructions automatically: {e}")

    print(f"\nComplete random data collection, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")


def main(task_name=None, task_config=None):
    task = class_decorator(task_name)
    args, embodiment_name = build_args(task_name, task_config)
    random_action_args = get_random_action_args(args)
    print_config(args, embodiment_name, random_action_args)
    run(task, args, random_action_args)


if __name__ == "__main__":
    from test_render import Sapien_TEST

    Sapien_TEST()

    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser = parser.parse_args()
    main(task_name=parser.task_name, task_config=parser.task_config)
