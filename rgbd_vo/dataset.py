from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class RGBDFrame:
    timestamp: float
    rgb_path: Path
    depth_path: Path


def read_tum_list(path: Path) -> list[tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rows.append((float(parts[0]), parts[1]))
    return rows


def associate_by_timestamp(
    first: list[tuple[float, str]],
    second: list[tuple[float, str]],
    max_difference: float = 0.02,
) -> list[tuple[float, str, float, str]]:
    second_unused = dict(second)
    matches: list[tuple[float, str, float, str]] = []

    for first_time, first_file in first:
        best_time = None
        best_diff = max_difference
        for second_time in list(second_unused.keys()):
            diff = abs(first_time - second_time)
            if diff < best_diff:
                best_time = second_time
                best_diff = diff

        if best_time is not None:
            matches.append((first_time, first_file, best_time, second_unused.pop(best_time)))

    matches.sort(key=lambda row: row[0])
    return matches


def write_associations(dataset_dir: Path, output_path: Path, max_difference: float = 0.02) -> int:
    rgb = read_tum_list(dataset_dir / "rgb.txt")
    depth = read_tum_list(dataset_dir / "depth.txt")
    matches = associate_by_timestamp(rgb, depth, max_difference=max_difference)

    with output_path.open("w", encoding="utf-8") as f:
        for rgb_time, rgb_file, depth_time, depth_file in matches:
            f.write(f"{rgb_time:.6f} {rgb_file} {depth_time:.6f} {depth_file}\n")
    return len(matches)


def load_associations(dataset_dir: Path, association_path: Path) -> list[RGBDFrame]:
    frames: list[RGBDFrame] = []
    with association_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            rgb_time = float(parts[0])
            rgb_file = parts[1]
            depth_file = parts[3]
            frames.append(
                RGBDFrame(
                    timestamp=rgb_time,
                    rgb_path=dataset_dir / rgb_file,
                    depth_path=dataset_dir / depth_file,
                )
            )
    return frames


def read_rgb_depth(frame: RGBDFrame) -> tuple[np.ndarray, np.ndarray]:
    rgb_bgr = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {frame.rgb_path}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth image: {frame.depth_path}")
    return rgb, depth

