from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


REPO_ROOT = Path(__file__).resolve().parent


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def infer_collection_dir(
    collection_dir: str | None,
    task_name: str | None,
    task_config: str | None,
    base_dir: str | None,
) -> Path:
    if collection_dir is not None:
        return resolve_project_path(collection_dir)

    if task_name is None or task_config is None:
        raise ValueError("Provide either --collection-dir or both --task-name and --task-config.")

    if base_dir is not None:
        return resolve_project_path(base_dir) / task_name / task_config

    task_cfg_path = REPO_ROOT / "task_config" / f"{task_config}.yml"
    if not task_cfg_path.exists():
        raise FileNotFoundError(f"Task config not found: {task_cfg_path}")
    if yaml is None:
        raise ImportError("PyYAML is required when resolving collection_dir from task_config/*.yml.")

    with open(task_cfg_path, "r", encoding="utf-8") as f:
        task_cfg = yaml.safe_load(f)

    save_path = task_cfg.get("save_path", "./data")
    return resolve_project_path(save_path) / task_name / task_config


def get_episode_paths(collection_dir: Path, episode: int | None, all_episodes: bool) -> list[Path]:
    data_dir = collection_dir / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Collected data directory not found: {data_dir}")

    if all_episodes:
        episode_paths = sorted(
            data_dir.glob("episode*.hdf5"),
            key=lambda path: int(path.stem.replace("episode", "")),
        )
        if not episode_paths:
            raise FileNotFoundError(f"No episode*.hdf5 files found in {data_dir}")
        return episode_paths

    if episode is None:
        raise ValueError("Either --episode or --all-episodes must be provided.")

    episode_path = data_dir / f"episode{episode}.hdf5"
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode file not found: {episode_path}")
    return [episode_path]


def count_video_frames(video_path: Path) -> int | None:
    if not video_path.exists():
        return None

    commands = [
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,nb_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ],
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ],
    ]

    for command in commands:
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

        for line in result.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                return int(line)

    return None


def load_episode_actions(hdf5_path: Path) -> dict[str, Any]:
    with h5py.File(hdf5_path, "r") as root:
        left_arm = root["/joint_action/left_arm"][()]
        left_gripper = root["/joint_action/left_gripper"][()]
        right_arm = root["/joint_action/right_arm"][()]
        right_gripper = root["/joint_action/right_gripper"][()]

        if "/joint_action/vector" in root:
            vector = root["/joint_action/vector"][()]
        else:
            vector = np.concatenate(
                [
                    left_arm,
                    left_gripper[:, None],
                    right_arm,
                    right_gripper[:, None],
                ],
                axis=1,
            )

    if vector.ndim != 2 or vector.shape[0] == 0:
        raise ValueError(f"Episode {hdf5_path} does not contain valid joint_action data.")

    action_frames = int(vector.shape[0])
    actions = []
    for frame_idx in range(action_frames):
        actions.append(
            {
                "frame_index": frame_idx,
                "left_arm": left_arm[frame_idx].astype(np.float64).tolist(),
                "left_gripper": float(left_gripper[frame_idx]),
                "right_arm": right_arm[frame_idx].astype(np.float64).tolist(),
                "right_gripper": float(right_gripper[frame_idx]),
                "vector": vector[frame_idx].astype(np.float64).tolist(),
            }
        )

    return {
        "action_frames": action_frames,
        "left_arm_dim": int(left_arm.shape[1]),
        "right_arm_dim": int(right_arm.shape[1]),
        "vector_dim": int(vector.shape[1]),
        "actions": actions,
    }


def inject_collection_metadata(payload: dict[str, Any], task_name: str | None, task_config: str | None) -> dict[str, Any]:
    enriched = dict(payload)
    if task_name is not None:
        enriched["task_name"] = task_name
    if task_config is not None:
        enriched["task_config"] = task_config
    return enriched


def export_episode(
    collection_dir: Path,
    hdf5_path: Path,
    output_dir: Path,
    task_name: str | None = None,
    task_config: str | None = None,
) -> dict[str, Any]:
    episode_name = hdf5_path.stem
    episode_idx = int(episode_name.replace("episode", ""))
    video_path = collection_dir / "video" / f"{episode_name}.mp4"
    json_path = output_dir / f"{episode_name}_actions.json"

    action_data = load_episode_actions(hdf5_path)
    video_frames = count_video_frames(video_path)

    payload = {
        "episode": episode_idx,
        "source_hdf5": str(hdf5_path.resolve()),
        "source_video": str(video_path.resolve()) if video_path.exists() else None,
        "action_frames": action_data["action_frames"],
        "video_frames": video_frames,
        "left_arm_dim": action_data["left_arm_dim"],
        "right_arm_dim": action_data["right_arm_dim"],
        "vector_dim": action_data["vector_dim"],
        "actions": action_data["actions"],
    }
    payload = inject_collection_metadata(payload, task_name, task_config)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        "episode": episode_idx,
        "json_path": str(json_path.resolve()),
        "video_path": str(video_path.resolve()) if video_path.exists() else None,
        "action_frames": action_data["action_frames"],
        "video_frames": video_frames,
        "frame_match": (video_frames == action_data["action_frames"]) if video_frames is not None else None,
        "task_name": task_name,
        "task_config": task_config,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export collected robot actions from episode HDF5 files into JSON and print frame counts."
    )
    parser.add_argument("--collection-dir", help="Collected task directory, e.g. ./data/adjust_bottle/wm_agilex_100")
    parser.add_argument("--base-dir", help="Optional base dir override, e.g. /data/agilex_random")
    parser.add_argument("--task-name", help="Task name, e.g. adjust_bottle")
    parser.add_argument("--task-config", help="Task config name without .yml, e.g. wm_agilex_100")
    parser.add_argument("--episode", type=int, help="Single episode index to export")
    parser.add_argument("--all-episodes", action="store_true", help="Export every episode under the collection dir")
    parser.add_argument(
        "--output-dir",
        help="Directory for exported JSON files. Defaults to <collection-dir>/action_json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collection_dir = infer_collection_dir(args.collection_dir, args.task_name, args.task_config, args.base_dir)
    episode_paths = get_episode_paths(collection_dir, args.episode, args.all_episodes)
    output_dir = resolve_project_path(args.output_dir) if args.output_dir else (collection_dir / "action_json")

    results = []
    for hdf5_path in episode_paths:
        result = export_episode(
            collection_dir,
            hdf5_path,
            output_dir,
            task_name=args.task_name,
            task_config=args.task_config,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    print(
        json.dumps(
            {
                "collection_dir": str(collection_dir.resolve()),
                "output_dir": str(output_dir.resolve()),
                "episodes": len(results),
                "task_name": args.task_name,
                "task_config": args.task_config,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
