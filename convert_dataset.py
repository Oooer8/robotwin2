#!/usr/bin/env python3
# 使用方法
# 修改143行agilex
"""
python convert_dataset.py \
    /root/workspace/robotwin_data \
    --output /root/workspace/robotwin_data/robotwin2_gtc_agilex_5k_eval

python convert_dataset.py \
    /root/workspace/robotwin_data \
    --output /root/workspace/robotwin_data/robotwin2_gtc_agilex_pi0 \
    --variant wm_agilex_100

python convert_dataset.py \
    /root/workspace/robotwin_data \
    --output /root/workspace/robotwin_data/robotwin2_gtc_agilex_5k_eval \
    --variant wm_agilex_100

python convert_dataset.py \
    /root/workspace/robotwin_data \
    --output /root/workspace/robotwin_data/robotwin2_gtc_agilex_5k_fail \
    --variant wm_agilex_100_fail

python convert_dataset.py \
    /root/workspace/robotwin_data \
    --output /root/workspace/robotwin_data/robotwin2_gtc_agilex_5k_fail_eval \
    --variant wm_agilex_100_fail
"""
"""
将 robotwin_data 数据集整理为 final 目录结构。

实际目录结构（注意有两层）：
  robotwin_data/agilex/<task>/<variant>/video/episodeN.mp4
                                       /instructions/episodeN.json
                                       /robot_only_replay/episodeN/{front,side,top}.mp4

编号规则：
  task_idx : 按照 TASK_ORDER 列表中的顺序索引（3位，从001开始）
  ep_idx   : 每个任务下 episode 的排序索引（3位）
  key      : task_idx + ep_idx，共6位，如 001002
"""

import os
import re
import json
import shutil
import argparse
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
# ── task_name -> task_idx (1-based) 的查找表 ────────────────────────────────
TASK_ORDER_MAP = {name: idx + 1 for idx, name in enumerate(TASK_ORDER)}


def extract_first_frame(video_path: Path, out_path: Path) -> bool:
    if not HAS_CV2:
        return False
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if ret:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), frame)
        return True
    print(f"  [WARN] 无法读取视频首帧: {video_path}")
    return False


def natural_sort_key(s: str):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def get_episode_index(name: str) -> int:
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else -1


def find_data_dir(task_dir: Path, variant: str | None = None) -> Path | None:
    """
    在任务目录下找到实际包含 video/instructions/robot_only_replay 的子目录。
    支持两种情况：
      1. task_dir 本身就包含这些子目录（无嵌套）
      2. task_dir/<variant>/ 下包含这些子目录（有一层嵌套，如 wm_agilex_100）

    参数：
      variant: 若指定，则精确匹配该名称的子目录；否则取第一个匹配的子目录。
    """
    target_subdirs = {"video", "instructions", "robot_only_replay"}

    # 情况1：task_dir 本身
    children = {d.name for d in task_dir.iterdir() if d.is_dir()}
    if target_subdirs & children:
        return task_dir

    # 情况2：往下再找一层
    candidates = sorted(
        [d for d in task_dir.iterdir() if d.is_dir() and not d.name.startswith('.')],
        key=lambda d: natural_sort_key(d.name)
    )

    if variant:
        # 精确匹配指定的 variant 目录名
        for sub in candidates:
            if sub.name == variant:
                sub_children = {d.name for d in sub.iterdir() if d.is_dir()}
                if target_subdirs & sub_children:
                    return sub
        # 未找到精确匹配，打印警告后回退到第一个可用子目录
        print(f"  [WARN] 未找到指定 variant '{variant}'，回退到第一个可用子目录")

    # 未指定 variant 或精确匹配失败，取第一个匹配的子目录
    for sub in candidates:
        sub_children = {d.name for d in sub.iterdir() if d.is_dir()}
        if target_subdirs & sub_children:
            return sub

    return None


def process(robotwin_root: Path, output_root: Path, variant: str | None = None):
    agilex_root = robotwin_root / "agilex"
    if not agilex_root.exists():
        raise FileNotFoundError(f"找不到 agilex 目录: {agilex_root}")

    if variant:
        print(f"指定 variant: {variant}")

    # ── 按 TASK_ORDER 顺序构建任务列表 ──────────────────────────────────────
    task_dirs = []
    unknown_dirs = []

    for task_name in TASK_ORDER:
        task_path = agilex_root / task_name
        if task_path.exists() and task_path.is_dir():
            task_dirs.append(task_path)
        else:
            print(f"[WARN] TASK_ORDER 中的任务目录不存在，跳过: {task_name}")

    # 检测目录中存在但不在 TASK_ORDER 里的任务（提示用户）
    for d in agilex_root.iterdir():
        if d.is_dir() and not d.name.startswith('.') and d.name not in TASK_ORDER_MAP:
            unknown_dirs.append(d.name)
    if unknown_dirs:
        print(f"[WARN] 以下目录不在 TASK_ORDER 中，将被忽略: {sorted(unknown_dirs)}")

    if not task_dirs:
        print("[WARN] 没有找到任何有效任务文件夹。")
        return

    # 创建输出目录
    for sub in ["head", "urdf_front", "urdf_side", "urdf_top"]:
        (output_root / "videos" / sub).mkdir(parents=True, exist_ok=True)

    prompts: dict[str, str] = {}
    copied_videos    = 0
    copied_replays   = 0
    extracted_frames = 0
    missing_files    = []

    for task_dir in task_dirs:
        task_idx  = TASK_ORDER_MAP[task_dir.name]
        task_code = f"{task_idx:03d}"

        # ── 找到实际数据目录（处理嵌套层，传入 variant）────────────────────
        data_dir = find_data_dir(task_dir, variant=variant)
        if data_dir is None:
            print(f"\n[Task {task_code}] {task_dir.name}  →  [WARN] 找不到数据子目录，跳过")
            continue

        variant_hint = f" ({data_dir.name})" if data_dir != task_dir else ""
        print(f"\n[Task {task_code}] {task_dir.name}{variant_hint}")

        video_dir       = data_dir / "video"
        instruction_dir = data_dir / "instructions"
        replay_dir      = data_dir / "robot_only_replay"

        # ── 收集 episode 编号 ─────────────────────────────────────────────
        episode_indices: set[int] = set()
        if video_dir.exists():
            for f in video_dir.iterdir():
                if f.suffix == ".mp4":
                    episode_indices.add(get_episode_index(f.stem))
        if instruction_dir.exists():
            for f in instruction_dir.iterdir():
                if f.suffix == ".json":
                    episode_indices.add(get_episode_index(f.stem))

        if not episode_indices:
            print(f"  [WARN] 未找到任何 episode，跳过。")
            continue

        for ep_num in sorted(episode_indices):
            ep_code = f"{ep_num:03d}"
            key     = f"{task_code}{ep_code}"
            ep_name = f"episode{ep_num}"
            print(f"  episode {ep_num:>3d}  →  key={key}")

            # ── 1. head 视频 ──────────────────────────────────────────────
            src_video = video_dir / f"{ep_name}.mp4"
            dst_video = output_root / "videos" / "head" / f"{key}.mp4"
            if src_video.exists():
                shutil.copy2(src_video, dst_video)
                copied_videos += 1
                # ── 1b. 首帧图片 ────────────────────────────────────────
                dst_jpg = output_root / "videos" / "head" / f"{key}.jpg"
                if extract_first_frame(src_video, dst_jpg):
                    extracted_frames += 1
            else:
                missing_files.append(str(src_video))
                print(f"    [MISS] head video: {src_video}")

            # ── 2. prompt ─────────────────────────────────────────────────
            src_json = instruction_dir / f"{ep_name}.json"
            if src_json.exists():
                try:
                    with open(src_json, encoding="utf-8") as f:
                        data = json.load(f)
                    seen = data.get("seen", [])
                    if seen:
                        prompts[key] = seen[0]
                    else:
                        print(f"    [WARN] {ep_name}.json 中 seen 列表为空")
                except Exception as e:
                    print(f"    [ERR] 读取 {src_json}: {e}")
            else:
                missing_files.append(str(src_json))
                print(f"    [MISS] instruction json: {src_json}")

            # ── 3. robot_only_replay ──────────────────────────────────────
            ep_replay_dir = replay_dir / ep_name
            for view in ("front", "side", "top"):
                src_replay = ep_replay_dir / f"{view}.mp4"
                dst_replay = output_root / "videos" / f"urdf_{view}" / f"{key}_{view}.mp4"
                if src_replay.exists():
                    shutil.copy2(src_replay, dst_replay)
                    copied_replays += 1
                else:
                    missing_files.append(str(src_replay))
                    print(f"    [MISS] replay {view}: {src_replay}")

    # ── 写入 prompts.json ─────────────────────────────────────────────────
    prompts_path = output_root / "prompts.json"
    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(prompts.items())), f, ensure_ascii=False, indent=2)

    # ── 汇总报告 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅  处理完成！")
    print(f"   输出目录       : {output_root.resolve()}")
    print(f"   使用 variant   : {variant if variant else '（自动选择）'}")
    print(f"   head 视频      : {copied_videos} 个")
    print(f"   首帧图片       : {extracted_frames} 张")
    print(f"   replay 视频    : {copied_replays} 个")
    print(f"   prompts 条目   : {len(prompts)} 条")
    if missing_files:
        print(f"\n⚠️  缺失文件（共 {len(missing_files)} 个）:")
        for p in missing_files:
            print(f"   - {p}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将 robotwin_data 整理为 final 目录结构"
    )
    parser.add_argument(
        "robotwin_data",
        type=str,
        help="robotwin_data 根目录路径"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出目录路径（默认：robotwin_data 同级的 final 目录）"
    )
    parser.add_argument(
        "--variant", "-v",
        type=str,
        default=None,
        help="指定变体子目录名称，例如 wm_agilex_100 或 wm_agilex_100_fail（精确匹配）"
    )
    args = parser.parse_args()

    robotwin_root = Path(args.robotwin_data).resolve()
    output_root   = Path(args.output).resolve() if args.output \
                    else robotwin_root.parent / "final"

    print(f"输入: {robotwin_root}")
    print(f"输出: {output_root}")
    process(robotwin_root, output_root, variant=args.variant)