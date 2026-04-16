#!/usr/bin/env python3
"""
Check frame-count consistency for videos produced by convert_dataset.py.

Default checked views:
  videos/head/<key>.mp4
  videos/left_camera/<key>_left_camera.mp4
  videos/right_camera/<key>_right_camera.mp4
  videos/urdf_front/<key>_front.mp4
  videos/urdf_head/<key>_head.mp4
  videos/urdf_side/<key>_side.mp4
  videos/urdf_top/<key>_top.mp4

Example:
  python check_converted_video_frames.py /path/to/robotwin_wow_clean_50_advanced
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ViewSpec:
    name: str
    directory: str
    suffix: str


@dataclass(frozen=True)
class FrameCount:
    frames: int | None
    source: str
    error: str | None = None


VIEW_SPECS = [
    ViewSpec("head", "head", ""),
    ViewSpec("left_camera", "left_camera", "_left_camera"),
    ViewSpec("right_camera", "right_camera", "_right_camera"),
    ViewSpec("urdf_front", "urdf_front", "_front"),
    ViewSpec("urdf_head", "urdf_head", "_head"),
    ViewSpec("urdf_side", "urdf_side", "_side"),
    ViewSpec("urdf_top", "urdf_top", "_top"),
]


def natural_sort_key(value: str) -> list[object]:
    return [int(item) if item.isdigit() else item.lower() for item in re.split(r"(\d+)", value)]


def resolve_dataset_paths(raw_root: str) -> tuple[Path, Path]:
    root = Path(raw_root).expanduser().resolve()
    if root.name == "videos":
        return root.parent, root
    return root, root / "videos"


def parse_key_from_file(path: Path, spec: ViewSpec) -> str | None:
    if path.suffix.lower() != ".mp4":
        return None

    stem = path.stem
    if not spec.suffix:
        return stem or None

    if not stem.endswith(spec.suffix):
        return None

    key = stem[: -len(spec.suffix)]
    return key or None


def expected_video_path(videos_dir: Path, key: str, spec: ViewSpec) -> Path:
    filename = f"{key}{spec.suffix}.mp4" if spec.suffix else f"{key}.mp4"
    return videos_dir / spec.directory / filename


def read_prompt_keys(dataset_root: Path) -> tuple[set[str], str | None]:
    prompts_path = dataset_root / "prompts.json"
    if not prompts_path.exists():
        return set(), None

    try:
        with open(prompts_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001 - report the concrete parser/read error.
        return set(), f"读取 prompts.json 失败: {exc}"

    if not isinstance(data, dict):
        return set(), "prompts.json 不是对象格式，已忽略"

    return {str(key) for key in data.keys()}, None


def build_video_index(
    videos_dir: Path,
    specs: list[ViewSpec],
) -> tuple[dict[str, dict[str, Path]], dict[str, int], list[str], list[str]]:
    index: dict[str, dict[str, Path]] = {}
    view_counts: dict[str, int] = {}
    missing_view_dirs: list[str] = []
    unexpected_files: list[str] = []

    for spec in specs:
        view_dir = videos_dir / spec.directory
        view_counts[spec.name] = 0

        if not view_dir.is_dir():
            missing_view_dirs.append(str(view_dir))
            continue

        for path in sorted(view_dir.iterdir(), key=lambda item: natural_sort_key(item.name)):
            if not path.is_file() or path.suffix.lower() != ".mp4":
                continue

            key = parse_key_from_file(path, spec)
            if key is None:
                unexpected_files.append(str(path))
                continue

            index.setdefault(key, {})[spec.name] = path
            view_counts[spec.name] += 1

    return index, view_counts, missing_view_dirs, unexpected_files


def parse_ffprobe_integer(raw: str) -> int | None:
    for line in raw.splitlines():
        value = line.strip()
        if not value or value == "N/A":
            continue
        try:
            return int(value)
        except ValueError:
            try:
                return int(float(value))
            except ValueError:
                continue
    return None


def run_ffprobe(ffprobe_bin: str, args: list[str], path: Path) -> tuple[int | None, str | None]:
    proc = subprocess.run(
        [ffprobe_bin, *args, str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        error = proc.stderr.strip() or f"ffprobe exited with code {proc.returncode}"
        return None, error

    frames = parse_ffprobe_integer(proc.stdout)
    if frames is None:
        return None, "ffprobe did not return an integer frame count"
    return frames, None


def count_frames(path: Path, ffprobe_bin: str) -> FrameCount:
    exact_args = [
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
    ]
    frames, exact_error = run_ffprobe(ffprobe_bin, exact_args, path)
    if frames is not None:
        return FrameCount(frames=frames, source="nb_read_frames")

    metadata_args = [
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
    ]
    frames, metadata_error = run_ffprobe(ffprobe_bin, metadata_args, path)
    if frames is not None:
        return FrameCount(frames=frames, source="nb_frames")

    error = exact_error or metadata_error or "unknown ffprobe error"
    return FrameCount(frames=None, source="ffprobe", error=error)


def count_all_frames(
    paths: list[Path],
    ffprobe_bin: str,
    workers: int,
) -> dict[Path, FrameCount]:
    if not paths:
        return {}

    results: dict[Path, FrameCount] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_path = {pool.submit(count_frames, path, ffprobe_bin): path for path in paths}
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                results[path] = future.result()
            except Exception as exc:  # noqa: BLE001 - keep checking other files.
                results[path] = FrameCount(frames=None, source="python", error=str(exc))
    return results


def relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def render_counts(counts: dict[str, int | None]) -> str:
    return " ".join(f"{spec.name}={counts.get(spec.name, 'NA')}" for spec in VIEW_SPECS)


def limited_rows(rows: list[dict[str, object]], max_details: int) -> list[dict[str, object]]:
    if max_details == 0:
        return rows
    return rows[:max_details]


def write_json_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 convert_dataset.py 输出的 3 个相机视角和 4 个 URDF 视角视频帧数是否一致"
    )
    parser.add_argument(
        "converted_dataset",
        help="convert 后的数据集根目录，也可以直接传其中的 videos 目录",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, max(1, (os.cpu_count() or 1) * 2)),
        help="并发调用 ffprobe 的线程数，默认最多 16",
    )
    parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        help="ffprobe 可执行文件路径或命令名，默认 ffprobe",
    )
    parser.add_argument(
        "--no-prompts",
        action="store_true",
        help="不要从 prompts.json 补充期望 key，只检查视频目录中出现过的 key",
    )
    parser.add_argument(
        "--show-ok",
        action="store_true",
        help="逐条打印帧数一致的 key；默认只打印异常详情和汇总",
    )
    parser.add_argument(
        "--max-details",
        type=int,
        default=50,
        help="每类异常最多打印多少条；0 表示全部，默认 50",
    )
    parser.add_argument(
        "--json-report",
        type=str,
        default=None,
        help="可选：把完整检查结果写到 JSON 文件",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        print("[ERROR] --workers 必须大于 0", file=sys.stderr)
        return 2
    if args.max_details < 0:
        print("[ERROR] --max-details 不能小于 0", file=sys.stderr)
        return 2

    ffprobe_path = shutil.which(args.ffprobe) or (
        args.ffprobe if Path(args.ffprobe).expanduser().exists() else None
    )
    if ffprobe_path is None:
        print("[ERROR] 未找到 ffprobe，请先安装 ffmpeg，或用 --ffprobe 指定路径。", file=sys.stderr)
        return 2

    dataset_root, videos_dir = resolve_dataset_paths(args.converted_dataset)
    if not videos_dir.is_dir():
        print(f"[ERROR] videos 目录不存在: {videos_dir}", file=sys.stderr)
        return 2

    video_index, view_counts, missing_view_dirs, unexpected_files = build_video_index(videos_dir, VIEW_SPECS)

    prompt_keys: set[str] = set()
    prompt_warning = None
    if not args.no_prompts:
        prompt_keys, prompt_warning = read_prompt_keys(dataset_root)

    all_keys = sorted(set(video_index.keys()) | prompt_keys, key=natural_sort_key)
    if not all_keys:
        print(f"[ERROR] 没有找到任何待检查的 key: {videos_dir}", file=sys.stderr)
        return 2

    paths_to_probe = sorted(
        {path for by_view in video_index.values() for path in by_view.values()},
        key=lambda path: natural_sort_key(str(path)),
    )

    print(f"[INFO] dataset : {dataset_root}")
    print(f"[INFO] videos  : {videos_dir}")
    print(f"[INFO] keys    : {len(all_keys)}")
    print(f"[INFO] videos  : {len(paths_to_probe)} mp4")
    print(f"[INFO] views   : {', '.join(spec.name for spec in VIEW_SPECS)}")
    print(f"[INFO] workers : {args.workers}")
    if prompt_warning:
        print(f"[WARN] {prompt_warning}")
    if missing_view_dirs:
        print(f"[WARN] 缺少视角目录: {', '.join(missing_view_dirs)}")
    if unexpected_files:
        print(f"[WARN] 有 {len(unexpected_files)} 个 mp4 文件名不符合当前视角命名规则，将不参与 key 对齐。")

    frame_counts = count_all_frames(paths_to_probe, str(ffprobe_path), args.workers)

    ok_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    read_error_rows: list[dict[str, object]] = []
    mismatch_rows: list[dict[str, object]] = []

    for key in all_keys:
        by_view = video_index.get(key, {})
        missing = [
            {
                "view": spec.name,
                "path": relpath(expected_video_path(videos_dir, key, spec), dataset_root),
            }
            for spec in VIEW_SPECS
            if spec.name not in by_view
        ]
        if missing:
            missing_rows.append({"key": key, "missing": missing})
            continue

        counts: dict[str, int | None] = {}
        errors: list[dict[str, str]] = []
        for spec in VIEW_SPECS:
            path = by_view[spec.name]
            result = frame_counts[path]
            counts[spec.name] = result.frames
            if result.frames is None:
                errors.append(
                    {
                        "view": spec.name,
                        "path": relpath(path, dataset_root),
                        "error": result.error or "unknown error",
                    }
                )

        if errors:
            read_error_rows.append({"key": key, "errors": errors})
            continue

        unique_counts = {count for count in counts.values() if count is not None}
        row = {"key": key, "counts": counts}
        if len(unique_counts) == 1:
            ok_rows.append(row)
            if args.show_ok:
                print(f"[OK]       {key}: {render_counts(counts)}")
        else:
            mismatch_rows.append(row)

    print("")
    for row in limited_rows(missing_rows, args.max_details):
        missing_text = ", ".join(
            f"{item['view']}={item['path']}" for item in row["missing"]  # type: ignore[index]
        )
        print(f"[MISSING]  {row['key']}: {missing_text}")
    if args.max_details and len(missing_rows) > args.max_details:
        print(f"[MISSING]  ... 还有 {len(missing_rows) - args.max_details} 条，使用 --max-details 0 查看全部")

    for row in limited_rows(read_error_rows, args.max_details):
        error_text = "; ".join(
            f"{item['view']}={item['path']} ({item['error']})" for item in row["errors"]  # type: ignore[index]
        )
        print(f"[READ_ERR] {row['key']}: {error_text}")
    if args.max_details and len(read_error_rows) > args.max_details:
        print(f"[READ_ERR] ... 还有 {len(read_error_rows) - args.max_details} 条，使用 --max-details 0 查看全部")

    for row in limited_rows(mismatch_rows, args.max_details):
        print(f"[MISMATCH] {row['key']}: {render_counts(row['counts'])}")  # type: ignore[arg-type]
    if args.max_details and len(mismatch_rows) > args.max_details:
        print(f"[MISMATCH] ... 还有 {len(mismatch_rows) - args.max_details} 条，使用 --max-details 0 查看全部")

    if unexpected_files and args.max_details == 0:
        for path in unexpected_files:
            print(f"[WARN] unexpected mp4 name: {relpath(Path(path), dataset_root)}")

    summary = {
        "keys_checked": len(all_keys),
        "ok": len(ok_rows),
        "missing": len(missing_rows),
        "read_errors": len(read_error_rows),
        "mismatches": len(mismatch_rows),
        "unexpected_files": len(unexpected_files),
        "view_file_counts": view_counts,
        "prompt_keys": len(prompt_keys),
    }

    print("")
    print("========== Summary ==========")
    print(f"keys checked     : {summary['keys_checked']}")
    print(f"consistent       : {summary['ok']}")
    print(f"missing files    : {summary['missing']}")
    print(f"read errors      : {summary['read_errors']}")
    print(f"frame mismatches : {summary['mismatches']}")
    print(f"unexpected names : {summary['unexpected_files']}")
    print("view file counts : " + ", ".join(f"{name}={count}" for name, count in view_counts.items()))

    report = {
        "dataset_root": str(dataset_root),
        "videos_dir": str(videos_dir),
        "views": [spec.name for spec in VIEW_SPECS],
        "summary": summary,
        "missing": missing_rows,
        "read_errors": read_error_rows,
        "mismatches": mismatch_rows,
        "unexpected_files": [relpath(Path(path), dataset_root) for path in unexpected_files],
    }
    if args.json_report:
        report_path = Path(args.json_report).expanduser()
        if not report_path.is_absolute():
            report_path = Path.cwd() / report_path
        write_json_report(report_path, report)
        print(f"json report      : {report_path}")

    failed = missing_rows or read_error_rows or mismatch_rows
    if failed:
        print("\n[FAIL] 存在缺失、读取失败或帧数不一致的视频。")
        return 1

    print("\n[OK] 7 个视角的视频帧数全部一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
