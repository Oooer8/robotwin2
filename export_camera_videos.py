import argparse
from pathlib import Path
import sys


CAMERA_VIDEO_SUFFIXES = {
    "head_camera": "",
    "left_camera": "_left_camera",
    "right_camera": "_right_camera",
}

CAMERA_ALIASES = {
    "head_camera": ("head_camera", "head", "cam_high", "front"),
    "left_camera": ("left_camera", "left_wrist", "left", "cam_left_wrist"),
    "right_camera": ("right_camera", "right_wrist", "right", "cam_right_wrist"),
}


def load_images_to_video():
    utils_path = str(Path(__file__).resolve().parent / "envs" / "utils")
    if utils_path not in sys.path:
        sys.path.append(utils_path)
    from images_to_video import images_to_video

    return images_to_video


def decode_rgb_frame(frame_bytes, cv2, np):
    if isinstance(frame_bytes, (bytes, bytearray, np.bytes_)):
        frame_buffer = frame_bytes
    elif isinstance(frame_bytes, np.ndarray):
        frame_buffer = frame_bytes.tobytes()
    else:
        raise TypeError(f"Unsupported frame buffer type: {type(frame_bytes)}")

    img = cv2.imdecode(np.frombuffer(frame_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode rgb frame")
    return img


def decode_rgb_dataset(dataset):
    import cv2
    import numpy as np

    return np.stack([decode_rgb_frame(dataset[i], cv2, np) for i in range(len(dataset))])


def read_rgb_dataset(dataset):
    import numpy as np

    if dataset.ndim == 4 and dataset.shape[-1] in (3, 4):
        frames = dataset[()]
        if frames.shape[-1] == 4:
            frames = frames[..., :3]
        if frames.dtype != np.uint8:
            frames = frames.clip(0, 255).astype(np.uint8)
        return frames

    if dataset.ndim == 3 and dataset.shape[-1] in (3, 4):
        frame = dataset[()]
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = frame.clip(0, 255).astype(np.uint8)
        return frame[None, ...]

    return decode_rgb_dataset(dataset)


def find_hdf5_files(path):
    path = Path(path)
    if path.is_file():
        return [path]

    hdf5_files = sorted(path.glob("episode*.hdf5"))
    if hdf5_files:
        return hdf5_files

    data_dir = path / "data"
    if data_dir.is_dir():
        return sorted(data_dir.glob("episode*.hdf5"))

    return sorted(path.rglob("episode*.hdf5"))


def default_video_dir(hdf5_path):
    if hdf5_path.parent.name == "data":
        return hdf5_path.parent.parent / "video"
    return hdf5_path.parent / "video"


def camera_video_path(video_dir, episode_name, camera_name):
    suffix = CAMERA_VIDEO_SUFFIXES[camera_name]
    return video_dir / f"{episode_name}{suffix}.mp4"


def iter_datasets(group, prefix=""):
    import h5py

    for name, item in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(item, h5py.Dataset):
            yield path, item
        elif isinstance(item, h5py.Group):
            yield from iter_datasets(item, path)


def dataset_summary(dataset):
    return f"shape={dataset.shape}, dtype={dataset.dtype}"


def looks_like_image_dataset(path, dataset):
    path_lower = path.lower()
    if dataset.ndim in (3, 4) and dataset.shape[-1] in (3, 4):
        return True
    if dataset.ndim >= 1 and dataset.dtype.kind in {"S", "O"}:
        return any(token in path_lower for token in ("rgb", "image", "camera", "color"))
    return False


def print_hdf5_overview(hdf5_path, max_items):
    import h5py

    with h5py.File(hdf5_path, "r") as root:
        print(f"\n[INSPECT] {hdf5_path}")
        print(f"Root keys: {list(root.keys())}")

        attrs = dict(root.attrs)
        if attrs:
            print(f"Root attrs: {attrs}")

        print("Datasets:")
        shown = 0
        image_candidates = []
        for path, dataset in iter_datasets(root):
            if shown < max_items:
                print(f"  /{path}: {dataset_summary(dataset)}")
            shown += 1
            if looks_like_image_dataset(path, dataset):
                image_candidates.append(path)

        if shown > max_items:
            print(f"  ... {shown - max_items} more datasets hidden")

        if image_candidates:
            print("Image-like candidates:")
            for path in image_candidates[:max_items]:
                print(f"  /{path}: {dataset_summary(root[path])}")
            if len(image_candidates) > max_items:
                print(f"  ... {len(image_candidates) - max_items} more candidates hidden")
        else:
            print("Image-like candidates: none")


def parse_camera_paths(raw_camera_paths):
    camera_paths = {}
    for item in raw_camera_paths or []:
        if "=" not in item:
            raise ValueError(f"Invalid --camera-paths entry: {item}. Expected camera=/hdf5/path")
        camera_name, hdf5_path = item.split("=", 1)
        camera_name = camera_name.strip()
        if camera_name not in CAMERA_VIDEO_SUFFIXES:
            valid = ", ".join(sorted(CAMERA_VIDEO_SUFFIXES))
            raise ValueError(f"Unknown camera name in --camera-paths: {camera_name}. Valid cameras: {valid}")
        camera_paths[camera_name] = hdf5_path.strip().strip("/")
    return camera_paths


def explicit_or_default_camera_paths(camera_name):
    paths = [
        f"observation/{camera_name}/rgb",
        f"observations/{camera_name}/rgb",
        f"observation/images/{camera_name}",
        f"observations/images/{camera_name}",
        f"images/{camera_name}",
        f"image/{camera_name}",
        f"rgb/{camera_name}",
        f"{camera_name}/rgb",
        f"{camera_name}_rgb",
        camera_name,
    ]
    return paths


def find_camera_dataset(root, camera_name, camera_paths):
    explicit_path = camera_paths.get(camera_name)
    if explicit_path is not None:
        if explicit_path in root:
            return explicit_path, root[explicit_path]
        raise KeyError(f"Configured path does not exist: /{explicit_path}")

    for path in explicit_or_default_camera_paths(camera_name):
        if path in root:
            return path, root[path]

    aliases = CAMERA_ALIASES[camera_name]
    matches = []
    for path, dataset in iter_datasets(root):
        path_lower = path.lower()
        if any(alias in path_lower for alias in aliases) and looks_like_image_dataset(path, dataset):
            matches.append((path, dataset))

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        candidates = ", ".join(f"/{path}" for path, _ in matches[:8])
        raise ValueError(
            f"Ambiguous image datasets for {camera_name}: {candidates}. "
            "Pass an explicit mapping with --camera-paths."
        )

    return None, None


def export_videos(hdf5_path, output_dir, cameras, fps, overwrite, camera_paths):
    import h5py

    images_to_video = load_images_to_video()
    hdf5_path = Path(hdf5_path)
    video_dir = Path(output_dir) if output_dir is not None else default_video_dir(hdf5_path)
    video_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    with h5py.File(hdf5_path, "r") as root:
        for camera_name in cameras:
            try:
                dataset_path, dataset = find_camera_dataset(root, camera_name, camera_paths)
            except (KeyError, ValueError) as exc:
                print(f"[MISS] {hdf5_path}: {camera_name}: {exc}")
                continue

            if dataset is None:
                print(f"[MISS] {hdf5_path}: {camera_name}: no matching RGB/image dataset")
                continue

            out_path = camera_video_path(video_dir, hdf5_path.stem, camera_name)
            if out_path.exists() and not overwrite:
                print(f"[SKIP] {out_path}")
                continue

            frames = read_rgb_dataset(dataset)
            images_to_video(frames, out_path=str(out_path), fps=fps)
            print(f"[OK] {hdf5_path}: /{dataset_path} -> {out_path}")
            exported += 1

    return exported


def main():
    parser = argparse.ArgumentParser(description="Export RoboTwin camera mp4 videos from episode*.hdf5 files.")
    parser.add_argument(
        "path",
        help="A single episode*.hdf5 file, a data directory, a collection directory, or a dataset root.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for exported videos. Defaults to sibling video/ dir.")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["head_camera", "left_camera", "right_camera"],
        choices=sorted(CAMERA_VIDEO_SUFFIXES),
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--camera-paths",
        nargs="*",
        default=[],
        metavar="CAMERA=/HDF5/PATH",
        help="Explicit HDF5 dataset mappings, e.g. head_camera=/foo/head_rgb left_camera=/foo/left_rgb.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print the first episode HDF5 structure and exit without exporting videos.",
    )
    parser.add_argument("--max-inspect-items", type=int, default=80)
    args = parser.parse_args()

    hdf5_files = find_hdf5_files(args.path)
    if not hdf5_files:
        raise FileNotFoundError(f"No episode*.hdf5 files found under {args.path}")

    print(f"Found {len(hdf5_files)} episode hdf5 files under {args.path}")
    if args.inspect:
        print_hdf5_overview(hdf5_files[0], args.max_inspect_items)
        return

    camera_paths = parse_camera_paths(args.camera_paths)
    total_exported = 0
    for hdf5_path in hdf5_files:
        total_exported += export_videos(hdf5_path, args.output_dir, args.cameras, args.fps, args.overwrite, camera_paths)

    print(f"Exported {total_exported} videos.")


if __name__ == "__main__":
    main()
