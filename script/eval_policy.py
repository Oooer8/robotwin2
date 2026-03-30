import sys
import os
import subprocess
import json

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb
from copy import deepcopy
from contextlib import contextmanager

try:
    import fcntl
except ImportError:
    fcntl = None

from generate_episode_instructions import *
from recover_episode_instructions import (
    generate_episode_payloads,
    recover_instructions,
    save_episode_descriptions,
)

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def resolve_collection_dir(base_save_path, task_name, task_config):
    save_root = Path(base_save_path)
    if not save_root.is_absolute():
        save_root = (Path(".") / save_root).resolve()
    return save_root / str(task_name) / str(task_config)


@contextmanager
def collection_lock(lock_path):
    if fcntl is None:
        yield
        return

    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def ensure_scene_info_file(save_path):
    info_file_path = save_path / "scene_info.json"
    if not info_file_path.exists():
        with open(info_file_path, "w", encoding="utf-8") as file:
            json.dump({}, file, ensure_ascii=False)
    return info_file_path


def load_seed_list(save_path):
    seed_path = save_path / "seed.txt"
    if not seed_path.exists():
        return []
    with open(seed_path, "r", encoding="utf-8") as file:
        seed_list = file.read().split()
    return [int(i) for i in seed_list]


def save_seed_list(save_path, seed_list):
    seed_path = save_path / "seed.txt"
    with open(seed_path, "w", encoding="utf-8") as file:
        for seed in seed_list:
            file.write(f"{seed} ")


def exist_hdf5(save_path, idx):
    file_path = save_path / "data" / f"episode{idx}.hdf5"
    return file_path.exists()


def ensure_eval_collection_state(args):
    if not bool(args.get("collect_data", False)):
        return None

    collection_dir = resolve_collection_dir(
        args.get("save_path", "./data"),
        args["task_name"],
        args["task_config"],
    )
    collection_dir.mkdir(parents=True, exist_ok=True)
    scene_info_path = ensure_scene_info_file(collection_dir)
    lock_path = collection_dir / ".collect.lock"
    counter_path = collection_dir / ".episode_counter"
    target_episode_num = int(args.get("episode_num", 100))

    with collection_lock(lock_path):
        seed_list = load_seed_list(collection_dir)

        episode_idx = 0
        while exist_hdf5(collection_dir, episode_idx):
            episode_idx += 1

        next_idx = episode_idx
        if counter_path.exists():
            try:
                next_idx = max(next_idx, int(counter_path.read_text(encoding="utf-8").strip()))
            except Exception:
                pass
        counter_path.write_text(str(next_idx), encoding="utf-8")

    return {
        "collection_dir": collection_dir,
        "scene_info_path": scene_info_path,
        "lock_path": lock_path,
        "counter_path": counter_path,
        "seed_list": seed_list,
        "next_episode_idx": next_idx,
        "target_episode_num": target_episode_num,
    }


def reserve_episode_idx(collection_state):
    if collection_state is None:
        return None

    with collection_lock(collection_state["lock_path"]):
        next_idx = 0
        if collection_state["counter_path"].exists():
            try:
                next_idx = int(collection_state["counter_path"].read_text(encoding="utf-8").strip())
            except Exception:
                next_idx = 0

        if next_idx >= int(collection_state["target_episode_num"]):
            return None

        collection_state["counter_path"].write_text(str(next_idx + 1), encoding="utf-8")
        collection_state["next_episode_idx"] = next_idx + 1
        return next_idx


def collection_has_capacity(collection_state):
    if collection_state is None:
        return True

    with collection_lock(collection_state["lock_path"]):
        next_idx = 0
        if collection_state["counter_path"].exists():
            try:
                next_idx = int(collection_state["counter_path"].read_text(encoding="utf-8").strip())
            except Exception:
                next_idx = 0
        return next_idx < int(collection_state["target_episode_num"])


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


def build_eval_episode_record(
    episode_info,
    seed,
    instruction,
    instruction_type,
    policy_name,
    ckpt_setting,
    success,
):
    if isinstance(episode_info, dict):
        record = deepcopy(episode_info)
    else:
        record = {"info": {}}

    if not isinstance(record.get("info"), dict):
        record["info"] = {}

    record["collector"] = "eval_policy"
    record["plan_success"] = True
    record["exec_success"] = bool(success)
    record["selected"] = True
    record["policy_success"] = bool(success)
    record["seed"] = int(seed)
    record["instruction"] = instruction
    record["instruction_type"] = instruction_type
    record["policy_name"] = policy_name
    record["ckpt_setting"] = ckpt_setting
    record["joint_storage_mode"] = {
        "joint_action": "real_joint_state",
        "joint_target": "drive_target",
    }
    return record


def persist_eval_episode(
    task_env,
    collection_state,
    episode_idx,
    episode_record,
    seed,
    task_name,
    max_instruction_num,
    clear_cache=False,
):
    info_db = None
    seed_list = None
    with collection_lock(collection_state["lock_path"]):
        with open(collection_state["scene_info_path"], "r", encoding="utf-8") as file:
            info_db = json.load(file)
        info_db[f"episode_{episode_idx}"] = episode_record
        with open(collection_state["scene_info_path"], "w", encoding="utf-8") as file:
            json.dump(info_db, file, ensure_ascii=False, indent=4)

        seed_list = load_seed_list(collection_state["collection_dir"])
        if len(seed_list) <= episode_idx:
            seed_list.extend([-1] * (episode_idx + 1 - len(seed_list)))
        seed_list[episode_idx] = int(seed)
        collection_state["seed_list"] = seed_list
        save_seed_list(collection_state["collection_dir"], collection_state["seed_list"])

    if info_db is not None and seed_list is not None:
        generated_descriptions = generate_episode_payloads(
            task_name=task_name,
            scene_info=info_db,
            max_num=max_instruction_num,
            seeds=seed_list,
        )
        generated_descriptions = [
            item for item in generated_descriptions
            if item["episode_index"] == episode_idx
        ]
        if generated_descriptions:
            save_episode_descriptions(collection_state["collection_dir"], generated_descriptions)

    task_env.close_env(clear_cache=clear_cache)
    task_env.merge_pkl_to_hdf5_video()
    safe_remove_data_cache(task_env)


def collection_is_complete(collection_state):
    if collection_state is None:
        return False

    target_episode_num = int(collection_state["target_episode_num"])
    data_dir = collection_state["collection_dir"] / "data"
    hdf5_count = len(list(data_dir.glob("episode*.hdf5"))) if data_dir.exists() else 0
    if hdf5_count < target_episode_num:
        return False

    seed_list = load_seed_list(collection_state["collection_dir"])
    if len(seed_list) < target_episode_num:
        return False
    return all(seed >= 0 for seed in seed_list[:target_episode_num])


def finalize_eval_collection(args, collection_state):
    if collection_state is None:
        return

    try:
        instruction_result = recover_instructions(
            collection_dir=collection_state["collection_dir"],
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


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["episode_num"] = int(usr_args.get("episode_num", args.get("episode_num", 100)))
    args["seed_stride"] = int(usr_args.get("seed_stride", 1))
    args["seed_start"] = usr_args.get("seed_start", None)

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

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
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    collection_state = ensure_eval_collection_state(args)
    if collection_state is not None:
        print(f"\033[94mEval Data Save Path:\033[0m {collection_state['collection_dir']}")

    use_eval_result_video = bool(args["eval_video_log"]) and collection_state is None
    if use_eval_result_video:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir
    else:
        args.pop("eval_video_save_dir", None)
        if bool(args["eval_video_log"]) and collection_state is not None:
            print("\033[94mEval Video Output:\033[0m disabled, using collect_data video export")

    # output camera config
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
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = int(args.get("episode_num", 100))
    topk = 1

    model = get_model(usr_args)
    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   collection_state=collection_state,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    if collection_state is None or collection_is_complete(collection_state):
        finalize_eval_collection(args, collection_state)
    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                collection_state=None,
                test_num=100,
                video_size=None,
                instruction_type=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = int(args["seed_start"]) if args.get("seed_start") is not None else st_seed
    seed_stride = int(args.get("seed_stride", 1))
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        if collection_state is not None and not collection_has_capacity(collection_state):
            break

        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += seed_stride
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                # stack_trace = traceback.format_exc()
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += seed_stride
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += seed_stride
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        episode_idx = now_id
        eval_args = args
        if collection_state is not None:
            episode_idx = reserve_episode_idx(collection_state)
            if episode_idx is None:
                break
            eval_args = deepcopy(args)
            eval_args["save_path"] = str(collection_state["collection_dir"])
            eval_args["save_data"] = True
            eval_args["store_real_joint_state"] = True
            eval_args["save_frame_limit"] = args.get("save_frame_limit", TASK_ENV.step_lim + 1)
            eval_args["enable_take_action_save"] = True

        TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=now_seed, is_test=True, **eval_args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        episode_record = None
        if collection_state is not None:
            episode_record = build_eval_episode_record(
                episode_info=episode_info,
                seed=now_seed,
                instruction=instruction,
                instruction_type=instruction_type,
                policy_name=args["policy_name"],
                ckpt_setting=args["ckpt_setting"],
                success=succ,
            )

        now_id += 1
        clear_cache = ((succ_seed + 1) % clear_cache_freq == 0)
        if collection_state is not None:
            persist_eval_episode(
                task_env=TASK_ENV,
                collection_state=collection_state,
                episode_idx=episode_idx,
                episode_record=episode_record,
                seed=now_seed,
                task_name=args["task_name"],
                max_instruction_num=int(args.get("language_num", 100)),
                clear_cache=clear_cache,
            )
        else:
            TASK_ENV.close_env(clear_cache=clear_cache)

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += seed_stride

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
