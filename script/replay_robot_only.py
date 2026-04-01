from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from envs._GLOBAL_CONFIGS import CONFIGS_PATH  # noqa: E402


@dataclass
class ReplayConfig:
    task_name: str
    task_config_name: str
    collection_dir: Path
    left_robot_file: Path
    right_robot_file: Path
    left_embodiment_config: dict[str, Any]
    right_embodiment_config: dict[str, Any]
    dual_arm_embodied: bool
    embodiment_dis: float | None
    replay_camera_type: str
    replay_camera_cfg: dict[str, Any]
    head_camera_type: str
    head_camera_cfg: dict[str, Any]
    head_camera_static_info: dict[str, Any] | None


class FfmpegVideoWriter:

    def __init__(self, output_path: Path, width: int, height: int, fps: float):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
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
                f"{self.width}x{self.height}",
                "-framerate",
                str(self.fps),
                "-i",
                "-",
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                "-crf",
                "23",
                str(self.output_path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Unexpected frame shape {frame.shape}, expected {(self.height, self.width, 3)}"
            )
        if frame.dtype != np.uint8:
            raise ValueError(f"Frame dtype must be uint8, got {frame.dtype}")
        assert self.process.stdin is not None
        self.process.stdin.write(frame.tobytes())

    def close(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with code {return_code} for {self.output_path}")


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def get_embodiment_file(embodiment_type: str, embodiment_types: dict[str, Any]) -> Path:
    if embodiment_type not in embodiment_types:
        raise KeyError(f"Unknown embodiment type: {embodiment_type}")
    robot_file = embodiment_types[embodiment_type]["file_path"]
    if robot_file is None:
        raise ValueError(f"Embodiment {embodiment_type} has no file_path configured")
    return resolve_project_path(robot_file)


def build_replay_config(task_name: str, task_config_name: str, collection_suffix: str | None = None) -> ReplayConfig:
    task_cfg_path = REPO_ROOT / "task_config" / f"{task_config_name}.yml"
    if not task_cfg_path.exists():
        raise FileNotFoundError(f"Task config not found: {task_cfg_path}")

    task_cfg = load_yaml(task_cfg_path)
    embodiment_types = load_yaml(Path(CONFIGS_PATH) / "_embodiment_config.yml")
    camera_types = load_yaml(Path(CONFIGS_PATH) / "_camera_config.yml")

    embodiment = task_cfg.get("embodiment")
    if not embodiment:
        raise ValueError(f"No embodiment configured in {task_cfg_path}")

    collection_folder = collection_suffix if collection_suffix else task_config_name
    collection_dir = resolve_project_path(task_cfg.get("save_path", "./data")) / task_name / collection_folder

    replay_camera_type = task_cfg["camera"]["wrist_camera_type"]
    if replay_camera_type not in camera_types:
        raise KeyError(f"Unknown camera type for replay: {replay_camera_type}")

    replay_camera_cfg = dict(camera_types[replay_camera_type])
    replay_camera_cfg.setdefault("near", 0.1)
    replay_camera_cfg.setdefault("far", 100.0)

    head_camera_type = task_cfg["camera"]["head_camera_type"]
    if head_camera_type not in camera_types:
        raise KeyError(f"Unknown head camera type for replay: {head_camera_type}")

    head_camera_cfg = dict(camera_types[head_camera_type])
    head_camera_cfg.setdefault("near", 0.1)
    head_camera_cfg.setdefault("far", 100.0)

    if len(embodiment) == 1:
        left_robot_file = get_embodiment_file(embodiment[0], embodiment_types)
        right_robot_file = get_embodiment_file(embodiment[0], embodiment_types)
        dual_arm_embodied = True
        embodiment_dis = None
    elif len(embodiment) == 3:
        left_robot_file = get_embodiment_file(embodiment[0], embodiment_types)
        right_robot_file = get_embodiment_file(embodiment[1], embodiment_types)
        dual_arm_embodied = False
        embodiment_dis = float(embodiment[2])
    else:
        raise ValueError("embodiment items should contain either 1 or 3 entries")

    left_embodiment_config = load_yaml(left_robot_file / "config.yml")
    right_embodiment_config = load_yaml(right_robot_file / "config.yml")
    head_camera_static_info = next(
        (
            dict(camera_info)
            for camera_info in left_embodiment_config.get("static_camera_list", [])
            if camera_info.get("name") == "head_camera"
        ),
        None,
    )

    return ReplayConfig(
        task_name=task_name,
        task_config_name=task_config_name,
        collection_dir=collection_dir,
        left_robot_file=left_robot_file,
        right_robot_file=right_robot_file,
        left_embodiment_config=left_embodiment_config,
        right_embodiment_config=right_embodiment_config,
        dual_arm_embodied=dual_arm_embodied,
        embodiment_dis=embodiment_dis,
        replay_camera_type=replay_camera_type,
        replay_camera_cfg=replay_camera_cfg,
        head_camera_type=head_camera_type,
        head_camera_cfg=head_camera_cfg,
        head_camera_static_info=head_camera_static_info,
    )


def get_episode_paths(collection_dir: Path, episode: int | None, all_episodes: bool) -> list[Path]:
    data_dir = collection_dir / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Collected episode directory not found: {data_dir}")

    if all_episodes:
        episode_paths = sorted(
            data_dir.glob("episode*.hdf5"),
            key=lambda path: int(path.stem.replace("episode", "")),
        )
        if not episode_paths:
            raise FileNotFoundError(f"No episode*.hdf5 files found in {data_dir}")
        return episode_paths

    if episode is None:
        raise ValueError("Either --episode or --all-episodes must be provided")

    episode_path = data_dir / f"episode{episode}.hdf5"
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode file not found: {episode_path}")
    return [episode_path]


def load_episode_states(hdf5_path: Path) -> tuple[np.ndarray, int, int]:
    import h5py

    with h5py.File(hdf5_path, "r") as root:
        left_arm = root["/joint_action/left_arm"][()]
        right_arm = root["/joint_action/right_arm"][()]

        if "/joint_action/vector" in root:
            states = root["/joint_action/vector"][()]
        else:
            left_gripper = root["/joint_action/left_gripper"][()]
            right_gripper = root["/joint_action/right_gripper"][()]
            states = np.concatenate(
                [
                    left_arm,
                    left_gripper[:, None],
                    right_arm,
                    right_gripper[:, None],
                ],
                axis=1,
            )

    if states.ndim != 2 or states.shape[0] == 0:
        raise ValueError(f"Episode {hdf5_path} does not contain valid joint states")

    return states.astype(np.float64), int(left_arm.shape[1]), int(right_arm.shape[1])


def load_head_camera_pose_from_hdf5(hdf5_path: Path) -> np.ndarray | None:
    import h5py

    with h5py.File(hdf5_path, "r") as root:
        dataset_path = "/observation/head_camera/cam2world_gl"
        if dataset_path not in root:
            return None

        cam2world = root[dataset_path][()]

    cam2world = np.asarray(cam2world, dtype=np.float64)
    if cam2world.shape == (4, 4):
        return cam2world
    if cam2world.ndim == 3 and cam2world.shape[1:] == (4, 4):
        if cam2world.shape[0] == 0:
            return None
        return cam2world[0]
    raise ValueError(
        f"Unexpected head camera pose shape in {hdf5_path}: {cam2world.shape}, expected (4, 4) or (T, 4, 4)"
    )


def create_scene() -> tuple[Any, Any, Any]:
    import sapien.core as sapien
    from sapien.render import set_global_config

    engine = sapien.Engine()
    set_global_config(max_num_materials=50000, max_num_textures=50000)
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    sapien.render.set_camera_shader_dir("rt")
    sapien.render.set_ray_tracing_samples_per_pixel(32)
    sapien.render.set_ray_tracing_path_depth(8)
    sapien.render.set_ray_tracing_denoiser("oidn")

    scene = engine.create_scene(sapien.SceneConfig())
    scene.set_timestep(1 / 250)
    scene.set_ambient_light([0.3, 0.3, 0.3])
    scene.add_directional_light([0, 0.6, -1], [1.2, 1.2, 1.2], shadow=True)
    scene.add_directional_light([0.5, -0.4, -1], [0.7, 0.7, 0.7], shadow=False)
    scene.add_point_light([1.6, -1.2, 2.2], [2.0, 2.0, 2.0], shadow=False)
    scene.add_point_light([-1.6, -1.2, 2.0], [1.5, 1.5, 1.5], shadow=False)
    return engine, renderer, scene


def create_robot(scene: Any, replay_cfg: ReplayConfig) -> Any:
    from envs.robot.robot import Robot

    robot_kwargs: dict[str, Any] = {
        "left_embodiment_config": replay_cfg.left_embodiment_config,
        "right_embodiment_config": replay_cfg.right_embodiment_config,
        "left_robot_file": str(replay_cfg.left_robot_file),
        "right_robot_file": str(replay_cfg.right_robot_file),
        "dual_arm_embodied": replay_cfg.dual_arm_embodied,
    }
    if replay_cfg.embodiment_dis is not None:
        robot_kwargs["embodiment_dis"] = replay_cfg.embodiment_dis

    robot = Robot(scene, need_topp=False, **robot_kwargs)
    robot.init_joints()
    robot.move_to_homestate()
    return robot


def build_entity_cache(robot: Any) -> dict[int, dict[str, Any]]:
    caches: dict[int, dict[str, Any]] = {}
    for entity in {robot.left_entity, robot.right_entity}:
        active_joints = entity.get_active_joints()
        caches[id(entity)] = {
            "entity": entity,
            "joint_to_index": {joint.get_name(): idx for idx, joint in enumerate(active_joints)},
        }
    return caches


def normalized_gripper_targets(robot: Any, arm_tag: str, gripper_val: float) -> list[tuple[Any, float]]:
    clipped = float(np.clip(gripper_val, 0.0, 1.0))
    if arm_tag == "left":
        joints = robot.left_gripper
        scale = robot.left_gripper_scale
        robot.left_gripper_val = clipped
    else:
        joints = robot.right_gripper
        scale = robot.right_gripper_scale
        robot.right_gripper_val = clipped

    real_gripper_val = scale[0] + clipped * (scale[1] - scale[0])
    return [(joint, real_gripper_val * multi + offset) for joint, multi, offset in joints]


def apply_robot_state(
    robot: Any,
    entity_cache: dict[int, dict[str, Any]],
    left_arm: np.ndarray,
    left_gripper: float,
    right_arm: np.ndarray,
    right_gripper: float,
) -> None:
    pending: dict[int, dict[str, Any]] = {}

    def ensure_pending(entity: Any) -> dict[str, Any]:
        entity_id = id(entity)
        if entity_id not in pending:
            pending[entity_id] = {
                "entity": entity,
                "qpos": entity.get_qpos().copy(),
                "qvel": np.zeros_like(entity.get_qvel()),
                "joint_to_index": entity_cache[entity_id]["joint_to_index"],
            }
        return pending[entity_id]

    def assign_joint(entity: Any, joint: Any, value: float) -> None:
        update = ensure_pending(entity)
        joint_idx = update["joint_to_index"][joint.get_name()]
        update["qpos"][joint_idx] = float(value)
        update["qvel"][joint_idx] = 0.0
        joint.set_drive_target(float(value))
        joint.set_drive_velocity_target(0.0)

    for joint, value in zip(robot.left_arm_joints, left_arm):
        assign_joint(robot.left_entity, joint, float(value))
    for joint, value in zip(robot.right_arm_joints, right_arm):
        assign_joint(robot.right_entity, joint, float(value))

    for joint, target in normalized_gripper_targets(robot, "left", left_gripper):
        assign_joint(robot.left_entity, joint, target)
    for joint, target in normalized_gripper_targets(robot, "right", right_gripper):
        assign_joint(robot.right_entity, joint, target)

    for update in pending.values():
        update["entity"].set_qpos(update["qpos"])
        update["entity"].set_qvel(update["qvel"])


def split_state(state: np.ndarray, left_arm_dim: int, right_arm_dim: int) -> tuple[np.ndarray, float, np.ndarray, float]:
    expected_dim = left_arm_dim + right_arm_dim + 2
    if state.shape[0] != expected_dim:
        raise ValueError(f"State dimension mismatch: got {state.shape[0]}, expected {expected_dim}")

    left_arm = state[:left_arm_dim]
    left_gripper = float(state[left_arm_dim])
    right_arm = state[left_arm_dim + 1:left_arm_dim + right_arm_dim + 1]
    right_gripper = float(state[left_arm_dim + right_arm_dim + 1])
    return left_arm, left_gripper, right_arm, right_gripper


def pose_from_axes(position: np.ndarray, forward: np.ndarray, left: np.ndarray) -> Any:
    import sapien.core as sapien

    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(forward, dtype=np.float64)
    left = np.asarray(left, dtype=np.float64)

    forward = forward / np.linalg.norm(forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    up = up / np.linalg.norm(up)

    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = position
    return sapien.Pose(mat44)


def build_pose_matrix(position: np.ndarray, forward: np.ndarray, left: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(forward, dtype=np.float64)
    left = np.asarray(left, dtype=np.float64)

    forward = forward / np.linalg.norm(forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    up = up / np.linalg.norm(up)

    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = position
    return mat44


def target_up_to_axes(position: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up_hint = np.asarray(up_hint, dtype=np.float64)

    forward = target - position
    forward = forward / np.linalg.norm(forward)

    left = np.cross(up_hint, forward)
    if np.linalg.norm(left) < 1e-6:
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        left = np.cross(fallback_up, forward)
    left = left / np.linalg.norm(left)
    return forward, left


def estimate_camera_setup(robot: Any, distance_scale: float) -> dict[str, dict[str, np.ndarray]]:
    links = list(robot.left_entity.get_links())
    if robot.right_entity is not robot.left_entity:
        links.extend(robot.right_entity.get_links())

    link_positions = np.array([link.get_pose().p for link in links], dtype=np.float64)
    mins = link_positions.min(axis=0)
    maxs = link_positions.max(axis=0)
    center = (mins + maxs) / 2.0

    span = np.maximum(maxs - mins, np.array([0.6, 0.6, 1.0], dtype=np.float64))
    radius = float(max(span.max(), 0.8))

    distance = radius * distance_scale

    camera_to_robot = 0.5 * distance
    front_target = center + np.array([0.0, 0.15 * distance, 0.1 * distance], dtype=np.float64)
    side_target = center + np.array([0.0, 0.15 * distance, 0.1 * distance], dtype=np.float64)
    top_target = center + np.array([0.0, 0.15 * distance, 0.1 * distance], dtype=np.float64)

    return {
        "front": {
            "position": front_target + np.array([0.0, camera_to_robot, 0.0], dtype=np.float64),
            "target": front_target,
            "up": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        },
        "side": {
            "position": side_target + np.array([camera_to_robot, 0.0, 0.0], dtype=np.float64),
            "target": side_target,
            "up": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        },
        "top": {
            "position": top_target + np.array([0.0, 0.0, camera_to_robot], dtype=np.float64),
            "target": top_target,
            "up": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        },
    }


def create_render_cameras(
    scene: Any,
    camera_setup: dict[str, dict[str, np.ndarray]],
    camera_cfg: dict[str, Any],
) -> dict[str, Any]:
    cameras = {}
    width = int(camera_cfg["w"])
    height = int(camera_cfg["h"])
    fovy = np.deg2rad(float(camera_cfg["fovy"]))
    near = float(camera_cfg.get("near", 0.1))
    far = float(camera_cfg.get("far", 100.0))

    for name, spec in camera_setup.items():
        forward, left = target_up_to_axes(spec["position"], spec["target"], spec["up"])
        camera = scene.add_camera(
            name=f"robot_only_{name}",
            width=width,
            height=height,
            fovy=fovy,
            near=near,
            far=far,
        )
        camera.entity.set_pose(pose_from_axes(spec["position"], forward, left))
        cameras[name] = camera
    return cameras


def create_pose_camera(
    scene: Any,
    name: str,
    pose_matrix: np.ndarray,
    camera_cfg: dict[str, Any],
) -> Any:
    import sapien.core as sapien

    width = int(camera_cfg["w"])
    height = int(camera_cfg["h"])
    fovy = np.deg2rad(float(camera_cfg["fovy"]))
    near = float(camera_cfg.get("near", 0.1))
    far = float(camera_cfg.get("far", 100.0))

    camera = scene.add_camera(
        name=name,
        width=width,
        height=height,
        fovy=fovy,
        near=near,
        far=far,
    )
    camera.entity.set_pose(sapien.Pose(np.asarray(pose_matrix, dtype=np.float64)))
    return camera


def get_fallback_head_camera_pose(replay_cfg: ReplayConfig) -> np.ndarray | None:
    camera_info = replay_cfg.head_camera_static_info
    if camera_info is None:
        return None

    position = np.asarray(camera_info["position"], dtype=np.float64)
    forward = camera_info.get("forward")
    if forward is None:
        forward = (-1.0 * position).tolist()
    left = camera_info.get("left")
    if left is None:
        left = [-forward[1], forward[0], 0.0]

    return build_pose_matrix(position, np.asarray(forward, dtype=np.float64), np.asarray(left, dtype=np.float64))


def render_camera_rgb(camera: Any) -> np.ndarray:
    camera.take_picture()
    rgba = camera.get_picture("Color")
    rgb = (rgba[:, :, :3] * 255.0).clip(0, 255).astype(np.uint8)
    return rgb


def build_output_dir(collection_dir: Path, episode_idx: int) -> Path:
    return collection_dir / "robot_only_replay" / f"episode{episode_idx}"


def replay_episode(
    replay_cfg: ReplayConfig,
    hdf5_path: Path,
    fps: float,
    distance_scale: float,
    max_frames: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    states, left_arm_dim, right_arm_dim = load_episode_states(hdf5_path)
    if max_frames is not None:
        states = states[:max_frames]

    episode_idx = int(hdf5_path.stem.replace("episode", ""))
    output_dir = build_output_dir(replay_cfg.collection_dir, episode_idx)
    view_paths = {view: output_dir / f"{view}.mp4" for view in ("front", "side", "top", "head")}
    manifest_path = output_dir / "manifest.json"

    if not overwrite and all(path.exists() for path in view_paths.values()) and manifest_path.exists():
        return {
            "episode": episode_idx,
            "output_dir": str(output_dir),
            "skipped": True,
        }

    _, _, scene = create_scene()
    robot = create_robot(scene, replay_cfg)
    entity_cache = build_entity_cache(robot)

    first_left_arm, first_left_gripper, first_right_arm, first_right_gripper = split_state(
        states[0], left_arm_dim, right_arm_dim
    )
    apply_robot_state(
        robot,
        entity_cache,
        first_left_arm,
        first_left_gripper,
        first_right_arm,
        first_right_gripper,
    )
    scene.step()
    scene.update_render()

    camera_setup = estimate_camera_setup(robot, distance_scale)
    cameras = create_render_cameras(scene, camera_setup, replay_cfg.replay_camera_cfg)
    head_camera_pose = load_head_camera_pose_from_hdf5(hdf5_path)
    head_camera_pose_source = "hdf5_observation.head_camera.cam2world_gl"
    if head_camera_pose is None:
        head_camera_pose = get_fallback_head_camera_pose(replay_cfg)
        head_camera_pose_source = "embodiment_static_camera_list"
    if head_camera_pose is None:
        raise ValueError(
            f"Unable to resolve head camera pose for {hdf5_path}: "
            "missing /observation/head_camera/cam2world_gl and no head_camera in embodiment static_camera_list"
        )
    cameras["head"] = create_pose_camera(
        scene,
        name="robot_only_head",
        pose_matrix=head_camera_pose,
        camera_cfg=replay_cfg.head_camera_cfg,
    )

    view_camera_cfgs = {
        "front": replay_cfg.replay_camera_cfg,
        "side": replay_cfg.replay_camera_cfg,
        "top": replay_cfg.replay_camera_cfg,
        "head": replay_cfg.head_camera_cfg,
    }
    writers = {
        view: FfmpegVideoWriter(path, int(view_camera_cfgs[view]["w"]), int(view_camera_cfgs[view]["h"]), fps)
        for view, path in view_paths.items()
    }

    try:
        for state in states:
            left_arm, left_gripper, right_arm, right_gripper = split_state(state, left_arm_dim, right_arm_dim)
            apply_robot_state(robot, entity_cache, left_arm, left_gripper, right_arm, right_gripper)
            scene.step()
            scene.update_render()
            for view, camera in cameras.items():
                writers[view].write(render_camera_rgb(camera))
    finally:
        close_error = None
        for writer in writers.values():
            try:
                writer.close()
            except Exception as exc:  # pragma: no cover
                close_error = exc
        if close_error is not None:
            raise close_error

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task_name": replay_cfg.task_name,
        "task_config": replay_cfg.task_config_name,
        "episode": episode_idx,
        "source_hdf5": str(hdf5_path.resolve()),
        "frames": int(states.shape[0]),
        "fps": fps,
        "camera_model": {
            "type": replay_cfg.replay_camera_type,
            "width": int(replay_cfg.replay_camera_cfg["w"]),
            "height": int(replay_cfg.replay_camera_cfg["h"]),
            "fovy_deg": float(replay_cfg.replay_camera_cfg["fovy"]),
            "near": float(replay_cfg.replay_camera_cfg.get("near", 0.1)),
            "far": float(replay_cfg.replay_camera_cfg.get("far", 100.0)),
        },
        "head_camera_model": {
            "type": replay_cfg.head_camera_type,
            "width": int(replay_cfg.head_camera_cfg["w"]),
            "height": int(replay_cfg.head_camera_cfg["h"]),
            "fovy_deg": float(replay_cfg.head_camera_cfg["fovy"]),
            "near": float(replay_cfg.head_camera_cfg.get("near", 0.1)),
            "far": float(replay_cfg.head_camera_cfg.get("far", 100.0)),
        },
        "views": {view: str(path.resolve()) for view, path in view_paths.items()},
        "camera_setup": {
            view: {
                key: value.tolist() for key, value in spec.items()
            }
            for view, spec in camera_setup.items()
        },
        "head_camera_pose_source": head_camera_pose_source,
        "head_camera_pose": head_camera_pose.tolist(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return {
        "episode": episode_idx,
        "output_dir": str(output_dir),
        "frames": int(states.shape[0]),
        "skipped": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render robot-only replay videos from RobotWin2 collected joint trajectories."
    )
    parser.add_argument("--task-name", required=True, help="Task name, e.g. beat_block_hammer")
    parser.add_argument("--task-config", required=True, help="Task config name without .yml, e.g. demo_clean")
    parser.add_argument("--episode", type=int, help="Single episode index to render")
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="Render every episode under the collected data directory",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS")
    parser.add_argument(
        "--distance-scale",
        type=float,
        default=1.8,
        help="Scale factor used to place the front/side/top cameras around the robot",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional debug limit for rendered frames")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing replay videos")
    parser.add_argument(
        "--collection-suffix",
        default=None,
        help="Override the collection dir name. e.g. wm_agilex_100_fail "
            "will read from <save_path>/<task>/<collection-suffix> "
            "instead of <save_path>/<task>/<task-config>",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay_cfg = build_replay_config(args.task_name, args.task_config, args.collection_suffix)


    for path in [replay_cfg.left_robot_file, replay_cfg.right_robot_file]:
        if not path.exists():
            raise FileNotFoundError(
                f"Robot embodiment assets not found: {path}. Please download/install the assets first."
            )

    episode_paths = get_episode_paths(replay_cfg.collection_dir, args.episode, args.all_episodes)
    results = []
    for hdf5_path in episode_paths:
        result = replay_episode(
            replay_cfg=replay_cfg,
            hdf5_path=hdf5_path,
            fps=args.fps,
            distance_scale=args.distance_scale,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    print(
        json.dumps(
            {
                "task_name": args.task_name,
                "task_config": args.task_config,
                "episodes": len(results),
                "collection_dir": str(replay_cfg.collection_dir.resolve()),
                "replay_dir": str((replay_cfg.collection_dir / "robot_only_replay").resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
