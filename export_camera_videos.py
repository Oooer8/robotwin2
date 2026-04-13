import argparse
from pathlib import Path
import sys


CAMERA_VIDEO_SUFFIXES = {
    "head_camera": "",
    "left_camera": "_left_camera",
    "right_camera": "_right_camera",
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


def export_videos(hdf5_path, output_dir, cameras, fps, overwrite):
    import h5py

    images_to_video = load_images_to_video()
    hdf5_path = Path(hdf5_path)
    video_dir = Path(output_dir) if output_dir is not None else default_video_dir(hdf5_path)
    video_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as root:
        observation = root["observation"]
        for camera_name in cameras:
            if camera_name not in observation or "rgb" not in observation[camera_name]:
                print(f"[MISS] {hdf5_path}: /observation/{camera_name}/rgb")
                continue

            out_path = camera_video_path(video_dir, hdf5_path.stem, camera_name)
            if out_path.exists() and not overwrite:
                print(f"[SKIP] {out_path}")
                continue

            frames = decode_rgb_dataset(observation[camera_name]["rgb"])
            images_to_video(frames, out_path=str(out_path), fps=fps)


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
    args = parser.parse_args()

    hdf5_files = find_hdf5_files(args.path)
    if not hdf5_files:
        raise FileNotFoundError(f"No episode*.hdf5 files found under {args.path}")

    print(f"Found {len(hdf5_files)} episode hdf5 files under {args.path}")
    for hdf5_path in hdf5_files:
        export_videos(hdf5_path, args.output_dir, args.cameras, args.fps, args.overwrite)


if __name__ == "__main__":
    main()
