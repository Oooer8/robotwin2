from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent
DESCRIPTION_ROOT = REPO_ROOT / "description"
TASK_INSTRUCTION_ROOT = DESCRIPTION_ROOT / "task_instruction"
OBJECT_DESCRIPTION_ROOT = DESCRIPTION_ROOT / "objects_description"


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

    with open(task_cfg_path, "r", encoding="utf-8") as f:
        task_cfg = yaml.safe_load(f)

    save_path = task_cfg.get("save_path", "./data")
    return resolve_project_path(save_path) / task_name / task_config


def load_task_cfg(task_config: str) -> dict[str, Any]:
    task_cfg_path = REPO_ROOT / "task_config" / f"{task_config}.yml"
    if not task_cfg_path.exists():
        raise FileNotFoundError(f"Task config not found: {task_cfg_path}")
    with open(task_cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_scene_info(collection_dir: Path) -> dict[str, Any]:
    scene_info_path = collection_dir / "scene_info.json"
    if not scene_info_path.exists():
        raise FileNotFoundError(f"scene_info.json not found: {scene_info_path}")
    with open(scene_info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_seed_list(collection_dir: Path) -> list[int]:
    seed_path = collection_dir / "seed.txt"
    if not seed_path.exists():
        return []
    with open(seed_path, "r", encoding="utf-8") as f:
        tokens = f.read().split()
    return [int(token) for token in tokens]


def load_task_instructions(task_name: str) -> dict[str, Any]:
    file_path = TASK_INSTRUCTION_ROOT / f"{task_name}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Task instruction template not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_placeholders(instruction: str) -> list[str]:
    return re.findall(r"{([^}]+)}", instruction)


def filter_instructions(instructions: list[str], episode_params: dict[str, str], rng: random.Random) -> list[str]:
    shuffled = list(instructions)
    rng.shuffle(shuffled)
    filtered = []

    stripped_episode_params = {key.strip("{}"): value for key, value in episode_params.items()}
    arm_params = {key for key in stripped_episode_params.keys() if len(key) == 1 and "a" <= key <= "z"}

    for instruction in shuffled:
        placeholders = set(extract_placeholders(instruction))
        if placeholders == set(stripped_episode_params.keys()) or (
            arm_params
            and placeholders.union(arm_params) == set(stripped_episode_params.keys())
            and not arm_params.intersection(placeholders)
        ):
            filtered.append(instruction)

    return filtered


def choose_object_description(value: str, split: str, rng: random.Random) -> str:
    json_path = OBJECT_DESCRIPTION_ROOT / f"{value}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Object description file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    choices = json_data.get(split, [])
    if not choices:
        choices = json_data.get("seen", [])
    if not choices:
        raise ValueError(f"No object descriptions found in {json_path}")
    return f"the {rng.choice(choices)}"


def replace_placeholders(
    instruction: str,
    episode_params: dict[str, str],
    split: str,
    rng: random.Random,
) -> str:
    stripped_episode_params = {key.strip("{}"): value for key, value in episode_params.items()}

    for key, value in stripped_episode_params.items():
        placeholder = "{" + key + "}"
        if "\\" in value or "/" in value:
            value = choose_object_description(value, split, rng)
        elif (OBJECT_DESCRIPTION_ROOT / f"{value}.json").exists():
            value = choose_object_description(value, split, rng)
        elif len(key) == 1 and "a" <= key <= "z":
            value = f"the {value} arm"
        instruction = instruction.replace(placeholder, value)

    return instruction


def extract_episode_info(scene_info: dict[str, Any]) -> list[dict[str, Any]]:
    episodes: list[tuple[int, dict[str, Any]]] = []
    for episode_key, episode_data in scene_info.items():
        if not episode_key.startswith("episode_"):
            continue
        episode_idx = int(episode_key.split("_")[-1])
        info = episode_data.get("info", {}) if isinstance(episode_data, dict) else {}
        episodes.append((episode_idx, info))
    episodes.sort(key=lambda item: item[0])
    return [{"episode_index": episode_idx, "info": info} for episode_idx, info in episodes]


def generate_episode_payloads(
    task_name: str,
    scene_info: dict[str, Any],
    max_num: int,
    seeds: list[int],
) -> list[dict[str, Any]]:
    task_data = load_task_instructions(task_name)
    seen_instructions = task_data.get("seen", [])
    unseen_instructions = task_data.get("unseen", [])
    episodes = extract_episode_info(scene_info)

    results = []
    for item in episodes:
        episode_index = item["episode_index"]
        episode_info = item["info"]
        seed = seeds[episode_index] if episode_index < len(seeds) else episode_index
        seen_rng = random.Random(f"{task_name}:{episode_index}:{seed}:seen")
        unseen_rng = random.Random(f"{task_name}:{episode_index}:{seed}:unseen")

        filtered_seen = filter_instructions(seen_instructions, episode_info, seen_rng)
        filtered_unseen = filter_instructions(unseen_instructions, episode_info, unseen_rng)

        seen_desc = [
            replace_placeholders(instruction, episode_info, "seen", seen_rng)
            for instruction in filtered_seen[:max_num]
        ]
        unseen_desc = [
            replace_placeholders(instruction, episode_info, "unseen", unseen_rng)
            for instruction in filtered_unseen[:max_num]
        ]

        results.append(
            {
                "episode_index": episode_index,
                "seen": seen_desc,
                "unseen": unseen_desc,
            }
        )

    return results


def save_episode_descriptions(collection_dir: Path, generated_descriptions: list[dict[str, Any]]) -> None:
    output_dir = collection_dir / "instructions"
    output_dir.mkdir(parents=True, exist_ok=True)

    for episode_desc in generated_descriptions:
        output_file = output_dir / f"episode{episode_desc['episode_index']}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seen": episode_desc.get("seen", []),
                    "unseen": episode_desc.get("unseen", []),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


def recover_instructions(
    collection_dir: Path,
    task_name: str,
    task_config: str,
    max_num: int,
) -> dict[str, Any]:
    scene_info = load_scene_info(collection_dir)
    seeds = load_seed_list(collection_dir)
    results = generate_episode_payloads(task_name, scene_info, max_num=max_num, seeds=seeds)
    save_episode_descriptions(collection_dir, results)
    return {
        "collection_dir": str(collection_dir.resolve()),
        "task_name": task_name,
        "task_config": task_config,
        "episodes": len(results),
        "instructions_dir": str((collection_dir / "instructions").resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover per-episode instruction files for an existing collected dataset."
    )
    parser.add_argument("--collection-dir", help="Collected task directory, e.g. /data/agilex_random/adjust_bottle/wm_agilex_100_random")
    parser.add_argument("--base-dir", help="Optional base dir override, e.g. /data/agilex_random")
    parser.add_argument("--task-name", help="Task name, e.g. adjust_bottle")
    parser.add_argument("--task-config", help="Task config name without .yml, e.g. wm_agilex_100_random")
    parser.add_argument("--max-num", type=int, default=None, help="Maximum instruction count per episode. Defaults to language_num in task config or 100.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task_name is None or args.task_config is None:
        if args.collection_dir is None:
            raise ValueError("Provide either --collection-dir plus task metadata, or --task-name and --task-config.")

    task_cfg = load_task_cfg(args.task_config) if args.task_config is not None else {}
    max_num = int(args.max_num if args.max_num is not None else task_cfg.get("language_num", 100))
    collection_dir = infer_collection_dir(args.collection_dir, args.task_name, args.task_config, args.base_dir)
    result = recover_instructions(
        collection_dir=collection_dir,
        task_name=args.task_name,
        task_config=args.task_config,
        max_num=max_num,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
