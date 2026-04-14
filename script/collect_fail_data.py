
import sys

sys.path.append("./")

import importlib
import json
import os
import pickle
import shutil
import time
import traceback
from argparse import ArgumentParser
from copy import deepcopy
from types import MethodType

import numpy as np
import yaml

from envs import *


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


def should_collect_episode(task_env):
    plan_success = bool(task_env.plan_success)
    exec_success = bool(task_env.check_success())
    selected = plan_success and (not exec_success)
    return selected, plan_success, exec_success


def get_noise_config(args):
    noise_cfg = deepcopy(args.get("trajectory_noise", {}))
    noise_cfg.setdefault("source_task_config", args["task_config"])
    noise_cfg.setdefault("target_task_config", None)
    noise_cfg.setdefault("output_suffix", "fail")
    noise_cfg.setdefault("random_seed", 0)
    noise_cfg.setdefault("noise_type", "random_walk")
    noise_cfg.setdefault("max_attempt_per_episode", 12)
    noise_cfg.setdefault("position_noise_std", 0.003)
    noise_cfg.setdefault("velocity_noise_std", 0.0)
    noise_cfg.setdefault("joint_bias_std", 0.003)
    noise_cfg.setdefault("joint_drift_std", 0.005)
    noise_cfg.setdefault("scale_growth", 1.2)
    noise_cfg.setdefault("joint_clip", float(np.pi))
    noise_cfg.setdefault("recompute_velocity", True)
    noise_cfg.setdefault("save_noisy_traj", True)
    noise_cfg.setdefault("walk_clip", 0.2)
    noise_cfg.setdefault("walk_step_delta_clip", 0.003)
    return noise_cfg


def get_source_and_target_save_path(args, noise_cfg):
    base_save_path = args["save_path"]
    source_task_config = noise_cfg["source_task_config"]
    source_save_path = os.path.join(base_save_path, str(args["task_name"]), source_task_config)

    target_task_config = noise_cfg.get("target_task_config")
    if not target_task_config:
        output_suffix = str(noise_cfg["output_suffix"]).strip()
        if output_suffix:
            target_task_config = f"{source_task_config}_{output_suffix}"
        else:
            target_task_config = source_task_config
    target_save_path = os.path.join(base_save_path, str(args["task_name"]), target_task_config)
    return source_save_path, target_save_path, target_task_config


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


def ensure_scene_info_file(save_path):
    info_file_path = os.path.join(save_path, "scene_info.json")
    if not os.path.exists(info_file_path):
        with open(info_file_path, "w", encoding="utf-8") as file:
            json.dump({}, file, ensure_ascii=False)
    return info_file_path


def load_scene_info(save_path):
    info_file_path = ensure_scene_info_file(save_path)
    with open(info_file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_scene_info(save_path, info_db):
    info_file_path = ensure_scene_info_file(save_path)
    with open(info_file_path, "w", encoding="utf-8") as file:
        json.dump(info_db, file, ensure_ascii=False, indent=4)


def load_traj_data(save_path, idx):
    file_path = os.path.join(save_path, "_traj_data", f"episode{idx}.pkl")
    with open(file_path, "rb") as f:
        traj_data = pickle.load(f)
    return traj_data


def save_traj_data(save_path, idx, traj_data):
    traj_dir = os.path.join(save_path, "_traj_data")
    os.makedirs(traj_dir, exist_ok=True)
    file_path = os.path.join(traj_dir, f"episode{idx}.pkl")
    save_pkl(file_path, traj_data)


def copy_instruction_file(source_save_path, source_episode_idx, target_save_path, target_episode_idx):
    source_file = os.path.join(source_save_path, "instructions", f"episode{source_episode_idx}.json")
    if not os.path.exists(source_file):
        return False

    target_dir = os.path.join(target_save_path, "instructions")
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, f"episode{target_episode_idx}.json")
    shutil.copy2(source_file, target_file)
    return True


def exist_hdf5(save_path, idx):
    file_path = os.path.join(save_path, "data", f"episode{idx}.hdf5")
    return os.path.exists(file_path)


def get_existing_source_episode_indices(info_db):
    used_source_indices = set()
    for _, episode_info in info_db.items():
        if not isinstance(episode_info, dict):
            continue
        source_episode_idx = episode_info.get("source_episode_idx")
        if source_episode_idx is not None:
            used_source_indices.add(int(source_episode_idx))
    return used_source_indices


def _to_numpy_array(value, fallback_dtype=np.float32):
    if value is None:
        return None
    return np.asarray(value, dtype=fallback_dtype)


def _recompute_velocity(position):
    if position.shape[0] <= 1:
        return np.zeros_like(position, dtype=np.float32)
    return np.gradient(position, 1 / 250, axis=0).astype(np.float32)


def _build_random_walk_offset(
    num_step,
    dim,
    start_offset,
    step_std,
    drift_std,
    bias_std,
    walk_clip,
    step_delta_clip,
    rng,
):
    if num_step <= 0:
        return np.zeros((0, dim), dtype=np.float32)
    if num_step == 1:
        single_step = start_offset + rng.normal(loc=0.0, scale=bias_std, size=(dim,)).astype(np.float32)
        return np.clip(single_step, -walk_clip, walk_clip).reshape(1, -1).astype(np.float32)

    walk = np.zeros((num_step, dim), dtype=np.float32)
    current = np.asarray(start_offset, dtype=np.float32).copy()
    step_drift = rng.normal(loc=0.0, scale=drift_std, size=(dim,)).astype(np.float32)
    segment_bias = rng.normal(loc=0.0, scale=bias_std, size=(dim,)).astype(np.float32)

    current = np.clip(current + segment_bias, -walk_clip, walk_clip)
    walk[0] = current

    for step_idx in range(1, num_step):
        random_step = rng.normal(loc=0.0, scale=step_std, size=(dim,)).astype(np.float32)
        delta = random_step + step_drift
        delta = np.clip(delta, -step_delta_clip, step_delta_clip)
        current = current + delta
        current = np.clip(current, -walk_clip, walk_clip)
        walk[step_idx] = current

    return walk.astype(np.float32)


def add_noise_to_plan_result(plan_result, noise_cfg, attempt_idx, rng, start_offset):
    noisy_result = deepcopy(plan_result)

    if not isinstance(noisy_result, dict):
        return noisy_result, start_offset
    if noisy_result.get("status") != "Success":
        return noisy_result, start_offset
    if "position" not in noisy_result:
        return noisy_result, start_offset

    position = _to_numpy_array(noisy_result["position"])
    if position is None or position.ndim != 2 or position.size == 0:
        return noisy_result, start_offset

    scale = float(noise_cfg["scale_growth"]) ** int(attempt_idx)
    walk_step_std = float(noise_cfg["position_noise_std"]) * scale
    velocity_noise_std = float(noise_cfg["velocity_noise_std"]) * scale
    joint_bias_std = float(noise_cfg["joint_bias_std"]) * scale
    joint_drift_std = float(noise_cfg["joint_drift_std"]) * scale
    joint_clip = float(noise_cfg["joint_clip"])
    walk_clip = float(noise_cfg["walk_clip"]) * scale
    step_delta_clip = float(noise_cfg["walk_step_delta_clip"]) * scale

    start_offset = np.asarray(start_offset, dtype=np.float32)
    if start_offset.shape != (position.shape[1],):
        start_offset = np.zeros((position.shape[1],), dtype=np.float32)

    walk_offset = _build_random_walk_offset(
        num_step=position.shape[0],
        dim=position.shape[1],
        start_offset=start_offset,
        step_std=walk_step_std,
        drift_std=joint_drift_std,
        bias_std=joint_bias_std,
        walk_clip=walk_clip,
        step_delta_clip=step_delta_clip,
        rng=rng,
    )

    noisy_position = position + walk_offset
    noisy_position = np.clip(noisy_position, -joint_clip, joint_clip).astype(np.float32)
    noisy_result["position"] = noisy_position

    if bool(noise_cfg["recompute_velocity"]):
        noisy_velocity = _recompute_velocity(noisy_position)
    else:
        velocity = _to_numpy_array(noisy_result.get("velocity"))
        if velocity is None or velocity.shape != noisy_position.shape:
            noisy_velocity = _recompute_velocity(noisy_position)
        else:
            velocity_noise = rng.normal(loc=0.0, scale=velocity_noise_std, size=velocity.shape).astype(np.float32)
            noisy_velocity = (velocity + velocity_noise).astype(np.float32)

    noisy_result["velocity"] = noisy_velocity
    return noisy_result, walk_offset[-1].astype(np.float32)


def add_noise_to_joint_path(joint_path, noise_cfg, attempt_idx, rng):
    noisy_joint_path = []
    end_offset = None

    for plan_result in joint_path:
        position = _to_numpy_array(plan_result.get("position")) if isinstance(plan_result, dict) else None
        if end_offset is None and position is not None and position.ndim == 2 and position.size > 0:
            end_offset = np.zeros((position.shape[1],), dtype=np.float32)

        noisy_plan_result, end_offset = add_noise_to_plan_result(
            plan_result=plan_result,
            noise_cfg=noise_cfg,
            attempt_idx=attempt_idx,
            rng=rng,
            start_offset=end_offset,
        )
        noisy_joint_path.append(noisy_plan_result)

    return noisy_joint_path


def add_noise_to_traj_data(traj_data, noise_cfg, attempt_idx, rng):
    noisy_traj_data = deepcopy(traj_data)
    noisy_traj_data["left_joint_path"] = add_noise_to_joint_path(
        noisy_traj_data.get("left_joint_path", []),
        noise_cfg,
        attempt_idx,
        rng,
    )
    noisy_traj_data["right_joint_path"] = add_noise_to_joint_path(
        noisy_traj_data.get("right_joint_path", []),
        noise_cfg,
        attempt_idx,
        rng,
    )
    return noisy_traj_data


def build_noise_metadata(noise_cfg, attempt_idx):
    scale = float(noise_cfg["scale_growth"]) ** int(attempt_idx)
    return {
        "attempt_idx": int(attempt_idx),
        "scale": float(scale),
        "noise_type": str(noise_cfg["noise_type"]),
        "position_noise_std": float(noise_cfg["position_noise_std"]) * scale,
        "velocity_noise_std": float(noise_cfg["velocity_noise_std"]) * scale,
        "joint_bias_std": float(noise_cfg["joint_bias_std"]) * scale,
        "joint_drift_std": float(noise_cfg["joint_drift_std"]) * scale,
        "walk_clip": float(noise_cfg["walk_clip"]) * scale,
        "walk_step_delta_clip": float(noise_cfg["walk_step_delta_clip"]) * scale,
        "joint_clip": float(noise_cfg["joint_clip"]),
        "recompute_velocity": bool(noise_cfg["recompute_velocity"]),
    }


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
    if not hasattr(task_env, "_original_get_obs_for_noisy_fail_real_state"):
        task_env._original_get_obs_for_noisy_fail_real_state = task_env.get_obs

    original_get_obs = task_env._original_get_obs_for_noisy_fail_real_state

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


def safe_close_env(task_env, clear_cache=False):
    try:
        task_env.close_env(clear_cache=clear_cache)
    except Exception:
        pass

    viewer = getattr(task_env, "viewer", None)
    if viewer is not None:
        try:
            viewer.close()
        except Exception:
            pass


def safe_remove_data_cache(task_env):
    folder_path = getattr(task_env, "folder_path", None)
    if not isinstance(folder_path, dict):
        return
    cache_path = folder_path.get("cache")
    if cache_path and os.path.exists(cache_path):
        try:
            task_env.remove_data_cache()
        except Exception:
            pass


def apply_cli_overrides(
    args,
    source_collection_suffix=None,
    target_collection_suffix=None,
    output_suffix=None,
    save_path=None,
):
    if save_path:
        args["save_path"] = save_path

    if source_collection_suffix or target_collection_suffix or output_suffix is not None:
        args.setdefault("trajectory_noise", {})

    if source_collection_suffix:
        args["trajectory_noise"]["source_task_config"] = source_collection_suffix
    if target_collection_suffix:
        args["trajectory_noise"]["target_task_config"] = target_collection_suffix
    if output_suffix is not None:
        args["trajectory_noise"]["output_suffix"] = output_suffix


def main(
    task_name=None,
    task_config=None,
    source_collection_suffix=None,
    target_collection_suffix=None,
    output_suffix=None,
    save_path=None,
):
    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    apply_cli_overrides(
        args,
        source_collection_suffix=source_collection_suffix,
        target_collection_suffix=target_collection_suffix,
        output_suffix=output_suffix,
        save_path=save_path,
    )

    args["task_name"] = task_name
    args["task_config"] = task_config

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_name):
        robot_file = _embodiment_types[embodiment_name]["file_path"]
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

    noise_cfg = get_noise_config(args)
    source_save_path, target_save_path, target_task_config = get_source_and_target_save_path(args, noise_cfg)

    args["embodiment_name"] = embodiment_name
    args["source_save_path"] = source_save_path
    args["target_save_path"] = target_save_path
    args["target_task_config"] = target_task_config

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

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\033[94mCollect Mode:\033[0m plan_success_execution_fail")
    print("\033[94mSource Data:\033[0m " + source_save_path)
    print("\033[94mTarget Data:\033[0m " + target_save_path)
    print("\033[94mNoise Type:\033[0m " + str(noise_cfg["noise_type"]))
    print("\033[94mNoise Attempts / Episode:\033[0m " + str(noise_cfg["max_attempt_per_episode"]))
    print("\033[94mWalk Step Std:\033[0m " + str(noise_cfg["position_noise_std"]))
    print("\033[94mWalk Drift Std:\033[0m " + str(noise_cfg["joint_drift_std"]))
    print("\033[94mWalk Clip:\033[0m " + str(noise_cfg["walk_clip"]))
    print("\033[94mStep Delta Clip:\033[0m " + str(noise_cfg["walk_step_delta_clip"]))
    print("\033[94mJoint Storage:\033[0m real_state -> /joint_action, target -> /joint_target")
    print("\n==================================")

    run(task, args, noise_cfg)


def run(task_env, args, noise_cfg):
    source_save_path = args["source_save_path"]
    target_save_path = args["target_save_path"]

    source_seed_list = load_seed_list(source_save_path)
    if not source_seed_list:
        raise FileNotFoundError(f"source seed.txt not found in {source_save_path}")

    os.makedirs(target_save_path, exist_ok=True)
    info_db = load_scene_info(target_save_path)
    used_source_indices = get_existing_source_episode_indices(info_db)

    collected_seed_list = load_seed_list(target_save_path)
    collected_num = 0
    while exist_hdf5(target_save_path, collected_num):
        collected_num += 1

    fail_num = 0
    clear_cache_freq = int(args["clear_cache_freq"])
    max_collect_num = min(int(args["episode_num"]), len(source_seed_list))

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")
    print("Collect Mode: \033[34mplan_success_execution_fail\033[0m")
    print(f"Source Episodes: \033[34m{len(source_seed_list)}\033[0m")
    print(f"Target Episodes Needed: \033[34m{max_collect_num}\033[0m")
    print(f"Already Collected: \033[34m{collected_num}\033[0m")

    if collected_num >= max_collect_num:
        print("Target dataset already complete.")
        return

    base_random_seed = int(noise_cfg["random_seed"])

    for source_episode_idx, seed in enumerate(source_seed_list):
        if collected_num >= max_collect_num:
            break
        if source_episode_idx in used_source_indices:
            continue

        traj_data = load_traj_data(source_save_path, source_episode_idx)
        episode_selected = False

        for attempt_idx in range(int(noise_cfg["max_attempt_per_episode"])):
            noise_seed = base_random_seed + source_episode_idx * 1000 + attempt_idx
            rng = np.random.default_rng(noise_seed)
            noisy_traj_data = add_noise_to_traj_data(traj_data, noise_cfg, attempt_idx, rng)
            noise_meta = build_noise_metadata(noise_cfg, attempt_idx)
            noise_meta["noise_seed"] = int(noise_seed)

            episode_args = deepcopy(args)
            episode_args["save_path"] = target_save_path
            episode_args["task_config"] = args["target_task_config"]
            episode_args["need_plan"] = False
            episode_args["save_data"] = True
            episode_args["render_freq"] = 0
            episode_args["allow_replay_exhaust_as_fail"] = True
            episode_args["left_joint_path"] = noisy_traj_data["left_joint_path"]
            episode_args["right_joint_path"] = noisy_traj_data["right_joint_path"]

            try:
                print(f"\033[34mTask name: {args['task_name']}\033[0m")
                print(
                    f"source episode {source_episode_idx} -> target episode {collected_num}, "
                    f"attempt {attempt_idx + 1}/{noise_cfg['max_attempt_per_episode']} "
                    f"(seed = {seed}, noise_seed = {noise_seed})"
                )
                task_env.setup_demo(now_ep_num=collected_num, seed=seed, **episode_args)
                patch_get_obs_with_real_joint_state(task_env)
                task_env.get_obs()
                task_env.set_path_lst(episode_args)
                info = task_env.play_once()

                replay_exhausted = bool(getattr(task_env, "replay_exhausted", False))
                selected, plan_success, exec_success = should_collect_episode(task_env)
                if selected:
                    task_env.merge_pkl_to_hdf5_video()
                    if bool(noise_cfg["save_noisy_traj"]):
                        save_traj_data(target_save_path, collected_num, noisy_traj_data)
                    instruction_copied = copy_instruction_file(
                        source_save_path,
                        source_episode_idx,
                        target_save_path,
                        collected_num,
                    )

                    info["plan_success"] = plan_success
                    info["exec_success"] = exec_success
                    info["collect_mode"] = "plan_success_execution_fail"
                    info["selected"] = selected
                    info["replay_exhausted"] = replay_exhausted
                    if replay_exhausted:
                        info["failure_reason"] = "replay_path_exhausted"
                    info["source_episode_idx"] = int(source_episode_idx)
                    info["source_task_config"] = noise_cfg["source_task_config"]
                    info["instruction_copied_from_source"] = bool(instruction_copied)
                    info["joint_storage_mode"] = {
                        "joint_action": "real_joint_state",
                        "joint_target": "drive_target",
                    }
                    info["noise"] = noise_meta
                    info_db[f"episode_{collected_num}"] = info
                    save_scene_info(target_save_path, info_db)

                    collected_seed_list.append(seed)
                    save_seed_list(target_save_path, collected_seed_list)

                    safe_remove_data_cache(task_env)
                    safe_close_env(task_env, clear_cache=((collected_num + 1) % clear_cache_freq == 0))

                    print(
                        f"selected noisy fail episode {collected_num}! "
                        f"(source = {source_episode_idx}, seed = {seed}, "
                        f"plan_success = {plan_success}, exec_success = {exec_success})"
                    )

                    collected_num += 1
                    used_source_indices.add(source_episode_idx)
                    episode_selected = True
                    break

                print(
                    f"attempt skipped: "
                    f"(source = {source_episode_idx}, seed = {seed}, "
                    f"plan_success = {plan_success}, exec_success = {exec_success})"
                )
                fail_num += 1
                safe_remove_data_cache(task_env)
                safe_close_env(task_env)
                time.sleep(0.2)
            except UnStableError as e:
                print(" -------------")
                print(
                    f"source episode {source_episode_idx} unstable on attempt {attempt_idx + 1}! "
                    f"(seed = {seed})"
                )
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                safe_remove_data_cache(task_env)
                safe_close_env(task_env)
                time.sleep(0.3)
            except Exception as e:
                print(" -------------")
                print(
                    f"source episode {source_episode_idx} failed on attempt {attempt_idx + 1}! "
                    f"(seed = {seed})"
                )
                print("Error: ", e)
                print(traceback.format_exc())
                print(" -------------")
                fail_num += 1
                safe_remove_data_cache(task_env)
                safe_close_env(task_env)
                time.sleep(0.5)

        if not episode_selected:
            used_source_indices.add(source_episode_idx)

    print(
        f"\nComplete noisy-fail collection, collected \033[92m{collected_num}\033[0m episodes, "
        f"failed/skipped \033[91m{fail_num}\033[0m attempts.\n"
    )

    if collected_num < max_collect_num:
        print(
            "Warning: source successful trajectories were exhausted before reaching target episode_num. "
            f"Collected {collected_num} / {max_collect_num}."
        )


if __name__ == "__main__":
    from test_render import Sapien_TEST

    Sapien_TEST()

    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument(
        "--source-collection-suffix",
        "--source-task-config",
        dest="source_collection_suffix",
        default=None,
        help="Source collection directory under <save_path>/<task_name>/, e.g. aloha-agilex_clean_50.",
    )
    parser.add_argument(
        "--target-collection-suffix",
        default=None,
        help="Exact target fail collection directory. Defaults to <source_collection_suffix>_<output_suffix>.",
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Suffix appended to the source collection when target collection is not set. Default: fail.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Parent directory containing task folders. Overrides save_path in task_config/<task_config>.yml.",
    )
    parser = parser.parse_args()

    main(
        task_name=parser.task_name,
        task_config=parser.task_config,
        source_collection_suffix=parser.source_collection_suffix,
        target_collection_suffix=parser.target_collection_suffix,
        output_suffix=parser.output_suffix,
        save_path=parser.save_path,
    )
