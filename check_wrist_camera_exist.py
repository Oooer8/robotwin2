"""
python check_wrist_camera_exist.py \
    /root/workspace/robotwin_data/agilex/adjust_bottle/wm_agilex_100/data/episode0.hdf5
"""

import argparse
import os

import cv2
import h5py
import numpy as np


def decode_rgb_frame(frame_bytes):
    if isinstance(frame_bytes, (bytes, bytearray)):
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    elif isinstance(frame_bytes, np.ndarray):
        arr = np.frombuffer(frame_bytes.tobytes(), dtype=np.uint8)
    else:
        raise TypeError(f"Unsupported frame buffer type: {type(frame_bytes)}")

    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode rgb frame")
    return img


def summarize_camera(camera_group):
    summary = {}
    for key, dataset in camera_group.items():
        if not isinstance(dataset, h5py.Dataset):
            continue

        item = {
            "shape": dataset.shape,
            "dtype": str(dataset.dtype),
        }

        if key == "rgb" and len(dataset) > 0:
            first_frame = decode_rgb_frame(dataset[0])
            item["decoded_shape"] = first_frame.shape

        summary[key] = item
    return summary


def main():
    parser = argparse.ArgumentParser(description="Check camera data inside a RoboTwin HDF5 file.")
    parser.add_argument("hdf_path", help="Path to episode*.hdf5")
    args = parser.parse_args()

    hdf_path = os.path.abspath(args.hdf_path)
    if not os.path.isfile(hdf_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf_path}")

    with h5py.File(hdf_path, "r") as root:
        if "observation" not in root:
            raise KeyError("Missing /observation group")

        observation = root["observation"]
        camera_names = [name for name, item in observation.items() if isinstance(item, h5py.Group)]

        print(f"HDF5: {hdf_path}")
        print(f"Cameras under /observation: {camera_names}")
        print()

        expected = ["head_camera", "left_camera", "right_camera"]
        for camera_name in expected:
            print(f"[{camera_name}] {'FOUND' if camera_name in observation else 'MISSING'}")
        print()

        for camera_name in camera_names:
            print(f"== {camera_name} ==")
            camera_group = observation[camera_name]
            summary = summarize_camera(camera_group)

            if not summary:
                print("  no datasets")
                print()
                continue

            for key, item in summary.items():
                line = f"  {key}: shape={item['shape']}, dtype={item['dtype']}"
                if "decoded_shape" in item:
                    line += f", decoded_first_frame_shape={item['decoded_shape']}"
                print(line)
            print()


if __name__ == "__main__":
    main()
