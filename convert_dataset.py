#!/usr/bin/env python3
"""
将 RobotWin 数据集整理为 final 目录结构。

常用示例：
  python convert_dataset.py /root/workspace/robotwin_data \
      --tasks-root robotwin/dataset \
      --variant aloha-agilex_clean_50 \
      --output /root/workspace/robotwin_data/robotwin2_gtc_agilex_clean_50

如果不传 --tasks-root，脚本会按顺序自动尝试：
  robotwin_data/robotwin/dataset
  robotwin_data/dataset
  robotwin_data/agilex_old
  robotwin_data/agilex
  robotwin_data

编号规则：
  task_idx : 按照 TASK_ORDER 列表中的顺序索引（3位，从001开始）
  ep_idx   : episode 编号（3位）
  key      : task_idx + ep_idx，共6位，如 002000

默认会自动复制每个 episode 已存在的视角：
  video/episodeN.mp4              -> videos/head/<key>.mp4
  video/episodeN_left_camera.mp4  -> videos/left_camera/<key>_left_camera.mp4
  video/episodeN_right_camera.mp4 -> videos/right_camera/<key>_right_camera.mp4
  robot_only_replay/episodeN/front.mp4 -> videos/urdf_front/<key>_front.mp4
  robot_only_replay/episodeN/head.mp4  -> videos/urdf_head/<key>_head.mp4
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[WARN] opencv-python 未安装，将跳过首帧图片提取。")
    print("       安装方法: pip install opencv-python")


# ── 固定任务顺序 ────────────────────────────────────────────────────────────
TASK_ORDER = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]
TASK_ORDER_MAP = {name: idx + 1 for idx, name in enumerate(TASK_ORDER)}

DATA_MARKER_SUBDIRS = {"video", "instructions", "robot_only_replay", "data", "_traj_data"}
AUTO_ROOT_CANDIDATES = [
    Path("robotwin") / "dataset",
    Path("dataset"),
    Path("agilex_old"),
    Path("agilex"),
    Path("."),
]


def natural_sort_key(s: str):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def get_episode_index(name: str) -> int:
    m = re.search(r"episode(\d+)|(\d+)", name)
    if not m:
        return -1
    return int(m.group(1) or m.group(2))


def parse_csv(raw: str | None) -> list[str] | None:
    """None/auto/all 表示自动发现；none 表示禁用。"""
    if raw is None:
        return None
    value = raw.strip()
    if value.lower() in {"", "auto", "all"}:
        return None
    if value.lower() == "none":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_task_names(raw: str | None) -> list[str] | None:
    if raw is None or not raw.strip():
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_episode_video_file(path: Path) -> tuple[int, str] | None:
    match = re.match(r"^episode(\d+)(?:_(.+))?$", path.stem)
    if not match:
        return None
    return int(match.group(1)), match.group(2) or "head"


def data_dir_has_markers(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any((path / subdir).is_dir() for subdir in DATA_MARKER_SUBDIRS)


def looks_like_task_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    try:
        children = [d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    except OSError:
        return False

    for child in children:
        if data_dir_has_markers(child):
            return True
        try:
            if any(data_dir_has_markers(sub) for sub in child.iterdir() if sub.is_dir()):
                return True
        except OSError:
            continue
    return False


def resolve_tasks_root(robotwin_root: Path, tasks_root: str | None) -> Path:
    if tasks_root:
        root = Path(tasks_root)
        if not root.is_absolute():
            root = robotwin_root / root
        root = root.resolve()
        if not root.exists():
            raise FileNotFoundError(f"找不到任务根目录: {root}")
        return root

    for candidate in AUTO_ROOT_CANDIDATES:
        root = (robotwin_root / candidate).resolve()
        if looks_like_task_root(root):
            return root

    checked = ", ".join(str((robotwin_root / c).resolve()) for c in AUTO_ROOT_CANDIDATES)
    raise FileNotFoundError(f"无法自动识别任务根目录，已尝试: {checked}")


def find_data_dir(
    task_dir: Path,
    variant: str | None = None,
    allow_variant_fallback: bool = False,
) -> Path | None:
    """
    找到实际包含 video/instructions/robot_only_replay/data 的目录。

    支持：
      1. task_dir 本身就是数据目录
      2. task_dir/<variant> 是数据目录，例如 aloha-agilex_clean_50
      3. 未指定 variant 时，自动选第一个包含数据标记的子目录
    """
    if data_dir_has_markers(task_dir):
        return task_dir

    candidates = sorted(
        [d for d in task_dir.iterdir() if d.is_dir() and not d.name.startswith(".")],
        key=lambda d: natural_sort_key(d.name),
    )

    if variant:
        matched = task_dir / variant
        if data_dir_has_markers(matched):
            return matched
        if not allow_variant_fallback:
            return None
        print(f"  [WARN] 未找到指定 variant '{variant}'，回退到第一个可用子目录")

    for subdir in candidates:
        if data_dir_has_markers(subdir):
            return subdir

    return None


def build_task_list(
    tasks_root: Path,
    task_names: list[str] | None,
    include_unknown_tasks: bool,
) -> list[tuple[Path, int]]:
    tasks: list[tuple[Path, int]] = []
    missing_known: list[str] = []

    if task_names is None:
        ordered_names = TASK_ORDER[:]
    else:
        ordered_names = task_names

    next_unknown_idx = len(TASK_ORDER) + 1
    for task_name in ordered_names:
        task_path = tasks_root / task_name
        if task_path.exists() and task_path.is_dir():
            task_idx = TASK_ORDER_MAP.get(task_name)
            if task_idx is None:
                task_idx = next_unknown_idx
                next_unknown_idx += 1
            tasks.append((task_path, task_idx))
        elif task_names is None:
            missing_known.append(task_name)
        else:
            print(f"[WARN] 指定任务目录不存在，跳过: {task_name}")

    if task_names is None and missing_known:
        for task_name in missing_known:
            print(f"[WARN] TASK_ORDER 中的任务目录不存在，跳过: {task_name}")

    if task_names is not None:
        return tasks

    known_or_selected = set(ordered_names)
    unknown_dirs = sorted(
        [
            d.name
            for d in tasks_root.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name not in known_or_selected
        ],
        key=natural_sort_key,
    )
    if unknown_dirs and include_unknown_tasks:
        for task_name in unknown_dirs:
            task_path = tasks_root / task_name
            task_idx = TASK_ORDER_MAP.get(task_name)
            if task_idx is None:
                task_idx = next_unknown_idx
                next_unknown_idx += 1
            tasks.append((task_path, task_idx))
    elif unknown_dirs:
        print(f"[WARN] 以下目录不在任务列表中，将被忽略: {unknown_dirs}")

    return tasks


def parse_video_view(path: Path, ep_num: int) -> str | None:
    parsed = parse_episode_video_file(path)
    if parsed is None:
        return None
    parsed_ep_num, view = parsed
    if parsed_ep_num != ep_num:
        return None
    return view


def video_source_path(video_dir: Path, ep_name: str, view: str) -> Path:
    file_name = f"{ep_name}.mp4" if view == "head" else f"{ep_name}_{view}.mp4"
    return video_dir / file_name


def collect_episode_indices(data_dir: Path) -> set[int]:
    episode_indices: set[int] = set()

    video_dir = data_dir / "video"
    if video_dir.exists():
        for path in video_dir.iterdir():
            if path.suffix == ".mp4" and path.stem.startswith("episode"):
                ep_idx = get_episode_index(path.stem)
                if ep_idx >= 0:
                    episode_indices.add(ep_idx)

    instruction_dir = data_dir / "instructions"
    if instruction_dir.exists():
        for path in instruction_dir.iterdir():
            if path.suffix == ".json" and path.stem.startswith("episode"):
                ep_idx = get_episode_index(path.stem)
                if ep_idx >= 0:
                    episode_indices.add(ep_idx)

    data_hdf5_dir = data_dir / "data"
    if data_hdf5_dir.exists():
        for path in data_hdf5_dir.iterdir():
            if path.suffix == ".hdf5" and path.stem.startswith("episode"):
                ep_idx = get_episode_index(path.stem)
                if ep_idx >= 0:
                    episode_indices.add(ep_idx)

    replay_dir = data_dir / "robot_only_replay"
    if replay_dir.exists():
        for path in replay_dir.iterdir():
            if path.is_dir() and path.name.startswith("episode"):
                ep_idx = get_episode_index(path.name)
                if ep_idx >= 0:
                    episode_indices.add(ep_idx)

    return episode_indices


def discover_video_files(video_dir: Path, ep_num: int) -> dict[str, Path]:
    if not video_dir.exists():
        return {}

    files: dict[str, Path] = {}
    for path in sorted(video_dir.glob(f"episode{ep_num}*.mp4"), key=lambda p: natural_sort_key(p.name)):
        view = parse_video_view(path, ep_num)
        if view is not None:
            files[view] = path
    return files


def discover_video_views(video_dir: Path) -> list[str]:
    if not video_dir.exists():
        return []

    views: set[str] = set()
    for path in video_dir.glob("episode*.mp4"):
        parsed = parse_episode_video_file(path)
        if parsed is not None:
            _, view = parsed
            views.add(view)
    return sorted(views, key=natural_sort_key)


def discover_replay_files(replay_episode_dir: Path) -> dict[str, Path]:
    if not replay_episode_dir.exists():
        return {}
    return {
        path.stem: path
        for path in sorted(replay_episode_dir.glob("*.mp4"), key=lambda p: natural_sort_key(p.name))
    }


def discover_replay_views(replay_dir: Path) -> list[str]:
    if not replay_dir.exists():
        return []

    views: set[str] = set()
    for path in replay_dir.glob("episode*/*.mp4"):
        views.add(path.stem)
    return sorted(views, key=natural_sort_key)


def select_views(discovered: dict[str, Path], requested_views: list[str] | None) -> list[str]:
    if requested_views is None:
        return sorted(discovered, key=natural_sort_key)
    return requested_views


def find_missing_episode_indices(episode_indices: set[int], expected_episodes: int | None) -> list[int]:
    if expected_episodes is not None:
        expected = set(range(expected_episodes))
    elif episode_indices:
        expected = set(range(max(episode_indices) + 1))
    else:
        expected = set()
    return sorted(expected - episode_indices)


def format_counts(counts: Counter[str]) -> str:
    if not counts:
        return "0"
    total = sum(counts.values())
    details = ", ".join(f"{view}={count}" for view, count in sorted(counts.items(), key=lambda x: natural_sort_key(x[0])))
    return f"{total} ({details})"


def copy_file(src: Path, dst: Path, *, dry_run: bool, overwrite: bool) -> bool:
    if dry_run:
        print(f"    [DRY] {src} -> {dst}")
        return True
    if dst.exists() and not overwrite:
        print(f"    [SKIP] 已存在: {dst}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def extract_first_frame(video_path: Path, out_path: Path, *, dry_run: bool, overwrite: bool) -> bool:
    if dry_run:
        print(f"    [DRY] extract first frame: {video_path} -> {out_path}")
        return True
    if out_path.exists() and not overwrite:
        print(f"    [SKIP] 首帧已存在: {out_path}")
        return False
    if not HAS_CV2:
        return False

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if ret:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), frame)
        return True
    print(f"    [WARN] 无法读取视频首帧: {video_path}")
    return False


def camera_video_output(output_root: Path, key: str, view: str) -> Path:
    file_name = f"{key}.mp4" if view == "head" else f"{key}_{view}.mp4"
    return output_root / "videos" / view / file_name


def camera_frame_output(output_root: Path, key: str, view: str) -> Path:
    file_name = f"{key}.jpg" if view == "head" else f"{key}_{view}.jpg"
    return output_root / "videos" / view / file_name


def replay_video_output(output_root: Path, key: str, view: str) -> Path:
    return output_root / "videos" / f"urdf_{view}" / f"{key}_{view}.mp4"


def load_prompt(instruction_path: Path) -> str | None:
    with open(instruction_path, encoding="utf-8") as f:
        data = json.load(f)

    seen = data.get("seen", [])
    if isinstance(seen, list) and seen:
        return str(seen[0])

    for key in ("instruction", "prompt", "task_description"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value

    return None


def process(
    robotwin_root: Path,
    output_root: Path,
    *,
    tasks_root_arg: str | None = None,
    variant: str | None = None,
    task_names: list[str] | None = None,
    include_unknown_tasks: bool = False,
    video_views: list[str] | None = None,
    replay_views: list[str] | None = None,
    extract_frame_views: list[str] | None = None,
    expected_episodes: int | None = None,
    allow_variant_fallback: bool = False,
    dry_run: bool = False,
    overwrite: bool = True,
) -> None:
    tasks_root = resolve_tasks_root(robotwin_root, tasks_root_arg)
    task_dirs = build_task_list(tasks_root, task_names, include_unknown_tasks)

    if variant:
        print(f"指定 variant: {variant}")
    print(f"任务根目录: {tasks_root}")

    if not task_dirs:
        print("[WARN] 没有找到任何有效任务文件夹。")
        return

    prompts: dict[str, str] = {}
    copied_videos: Counter[str] = Counter()
    copied_replays: Counter[str] = Counter()
    extracted_frames: Counter[str] = Counter()
    missing_files: list[str] = []
    missing_episode_reports: list[dict[str, object]] = []
    skipped_tasks = 0

    extract_all_frames = extract_frame_views is None
    extract_frame_set = set(extract_frame_views or [])

    for task_dir, task_idx in task_dirs:
        task_code = f"{task_idx:03d}"

        data_dir = find_data_dir(
            task_dir,
            variant=variant,
            allow_variant_fallback=allow_variant_fallback,
        )
        if data_dir is None:
            print(f"\n[Task {task_code}] {task_dir.name}  ->  [WARN] 找不到数据子目录，跳过")
            skipped_tasks += 1
            continue

        variant_hint = f" ({data_dir.name})" if data_dir != task_dir else ""
        print(f"\n[Task {task_code}] {task_dir.name}{variant_hint}")

        video_dir = data_dir / "video"
        instruction_dir = data_dir / "instructions"
        replay_dir = data_dir / "robot_only_replay"

        episode_indices = collect_episode_indices(data_dir)
        if not episode_indices:
            print("  [WARN] 未找到任何 episode，跳过。")
            continue

        task_video_views = video_views if video_views is not None else discover_video_views(video_dir)
        task_replay_views = replay_views if replay_views is not None else discover_replay_views(replay_dir)
        missing_episode_indices = find_missing_episode_indices(episode_indices, expected_episodes)
        for missing_ep_num in missing_episode_indices:
            ep_code = f"{missing_ep_num:03d}"
            key = f"{task_code}{ep_code}"
            ep_name = f"episode{missing_ep_num}"
            expected_video_files = [
                str(video_source_path(video_dir, ep_name, view))
                for view in task_video_views
            ]
            expected_replay_files = [
                str(replay_dir / ep_name / f"{view}.mp4")
                for view in task_replay_views
            ]
            expected_instruction_file = str(instruction_dir / f"{ep_name}.json")
            missing_episode_reports.append(
                {
                    "task": task_dir.name,
                    "task_code": task_code,
                    "episode": missing_ep_num,
                    "key": key,
                    "video_files": expected_video_files,
                    "replay_files": expected_replay_files,
                    "instruction_file": expected_instruction_file,
                }
            )
            print(f"  [MISS_EP] episode {missing_ep_num:>3d}  ->  key={key}")

        for ep_num in sorted(episode_indices):
            ep_code = f"{ep_num:03d}"
            key = f"{task_code}{ep_code}"
            ep_name = f"episode{ep_num}"
            print(f"  episode {ep_num:>3d}  ->  key={key}")

            # 1. 真实采集/仿真 observation 视频，多视角自动发现。
            discovered_video_files = discover_video_files(video_dir, ep_num)
            for view in task_video_views:
                src_video = discovered_video_files.get(view)
                if src_video is None:
                    expected = video_source_path(video_dir, ep_name, view)
                    missing_files.append(str(expected))
                    print(f"    [MISS] video {view}: {expected}")
                    continue

                dst_video = camera_video_output(output_root, key, view)
                if copy_file(src_video, dst_video, dry_run=dry_run, overwrite=overwrite):
                    copied_videos[view] += 1

                if extract_all_frames or view in extract_frame_set:
                    dst_jpg = camera_frame_output(output_root, key, view)
                    if extract_first_frame(src_video, dst_jpg, dry_run=dry_run, overwrite=overwrite):
                        extracted_frames[view] += 1

            # 2. prompt。
            src_json = instruction_dir / f"{ep_name}.json"
            if src_json.exists():
                try:
                    prompt = load_prompt(src_json)
                    if prompt:
                        prompts[key] = prompt
                    else:
                        print(f"    [WARN] {ep_name}.json 中没有可用 prompt")
                except Exception as exc:
                    print(f"    [ERR] 读取 {src_json}: {exc}")
            else:
                missing_files.append(str(src_json))
                print(f"    [MISS] instruction json: {src_json}")

            # 3. robot_only_replay / URDF 视频，多视角自动发现。
            ep_replay_dir = replay_dir / ep_name
            discovered_replay_files = discover_replay_files(ep_replay_dir)
            for view in task_replay_views:
                src_replay = discovered_replay_files.get(view)
                if src_replay is None:
                    expected = ep_replay_dir / f"{view}.mp4"
                    missing_files.append(str(expected))
                    print(f"    [MISS] replay {view}: {expected}")
                    continue

                dst_replay = replay_video_output(output_root, key, view)
                if copy_file(src_replay, dst_replay, dry_run=dry_run, overwrite=overwrite):
                    copied_replays[view] += 1

    prompts_path = output_root / "prompts.json"
    if dry_run:
        print(f"\n[DRY] write prompts: {prompts_path} ({len(prompts)} 条)")
    else:
        prompts_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump(dict(sorted(prompts.items())), f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("处理完成")
    print(f"   输出目录       : {output_root.resolve()}")
    print(f"   使用 variant   : {variant if variant else '自动选择'}")
    print(f"   跳过任务       : {skipped_tasks} 个")
    print(f"   video 视频     : {format_counts(copied_videos)}")
    print(f"   首帧图片       : {format_counts(extracted_frames)}")
    print(f"   replay 视频    : {format_counts(copied_replays)}")
    print(f"   prompts 条目   : {len(prompts)} 条")
    if missing_episode_reports:
        print(f"\n缺失 episode（共 {len(missing_episode_reports)} 个）:")
        for item in missing_episode_reports:
            print(
                f"   - Task {item['task_code']} {item['task']}: "
                f"episode{item['episode']} -> key={item['key']}"
            )
            print(f"     instruction: {item['instruction_file']}")
            video_files = item["video_files"]
            replay_files = item["replay_files"]
            if video_files:
                print("     video:")
                for path in video_files:
                    print(f"       - {path}")
            if replay_files:
                print("     replay:")
                for path in replay_files:
                    print(f"       - {path}")
    if missing_files:
        print(f"\n缺失文件（共 {len(missing_files)} 个）:")
        for path in missing_files:
            print(f"   - {path}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="将 RobotWin 数据集整理为 final 目录结构")
    parser.add_argument("robotwin_data", type=str, help="robotwin_data 根目录路径")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="输出目录路径（默认：robotwin_data 同级的 final 目录）",
    )
    parser.add_argument(
        "--tasks-root",
        "-t",
        type=str,
        default=None,
        help="任务目录根路径；可传绝对路径，也可传相对 robotwin_data 的路径，例如 robotwin/dataset",
    )
    parser.add_argument(
        "--variant",
        "-v",
        type=str,
        default=None,
        help="指定任务下的数据子目录，例如 aloha-agilex_clean_50（精确匹配）",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="只处理指定任务，逗号分隔，例如 beat_block_hammer,lift_pot",
    )
    parser.add_argument(
        "--include-unknown-tasks",
        action="store_true",
        help="同时处理不在 TASK_ORDER 中的任务目录；编号排在 TASK_ORDER 之后",
    )
    parser.add_argument(
        "--video-views",
        type=str,
        default="auto",
        help="video 目录下要复制的视角，逗号分隔；默认 auto 表示复制已存在的全部视角",
    )
    parser.add_argument(
        "--replay-views",
        type=str,
        default="auto",
        help="robot_only_replay 下要复制的视角，逗号分隔；默认 auto 表示复制已存在的全部视角",
    )
    parser.add_argument(
        "--extract-frame-views",
        type=str,
        default="head",
        help="要提取首帧 jpg 的 video 视角，逗号分隔；可用 all 或 none，默认 head",
    )
    parser.add_argument(
        "--expected-episodes",
        type=int,
        default=None,
        help="每个任务期望的 episode 数，例如 50；用于检测末尾也缺失的 episode",
    )
    parser.add_argument(
        "--allow-variant-fallback",
        action="store_true",
        help="指定 variant 不存在时，允许回退到该任务下第一个可用数据目录",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要复制的文件，不实际写入",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="目标文件已存在时跳过，不覆盖",
    )
    parser.set_defaults(overwrite=True)
    args = parser.parse_args()

    robotwin_root = Path(args.robotwin_data).resolve()
    output_root = Path(args.output).resolve() if args.output else robotwin_root.parent / "final"

    print(f"输入: {robotwin_root}")
    print(f"输出: {output_root}")

    if args.expected_episodes is not None and args.expected_episodes < 0:
        raise ValueError("--expected-episodes 不能小于 0")

    process(
        robotwin_root,
        output_root,
        tasks_root_arg=args.tasks_root,
        variant=args.variant,
        task_names=parse_task_names(args.tasks),
        include_unknown_tasks=args.include_unknown_tasks,
        video_views=parse_csv(args.video_views),
        replay_views=parse_csv(args.replay_views),
        extract_frame_views=parse_csv(args.extract_frame_views),
        expected_episodes=args.expected_episodes,
        allow_variant_fallback=args.allow_variant_fallback,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
