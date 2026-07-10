import argparse
import sys
from pathlib import Path

import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgbd_vo.camera import intrinsics_from_name
from rgbd_vo.dataset import load_associations, read_rgb_depth
from rgbd_vo.pointcloud import resize_rgb_to_depth, rgbd_to_pointcloud, save_cloud


def main() -> None:
    parser = argparse.ArgumentParser(description="View one RGB-D frame as colored point cloud.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--association", required=True, type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--camera", default="freiburg1")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    frames = load_associations(args.dataset, args.association)
    frame = frames[args.index]
    intr = intrinsics_from_name(args.camera)

    rgb, depth_raw = read_rgb_depth(frame)
    rgb = resize_rgb_to_depth(rgb, depth_raw)
    cloud = rgbd_to_pointcloud(rgb, depth_raw, intr, stride=args.stride)

    print(f"Frame timestamp: {frame.timestamp:.6f}")
    print(f"Point count: {len(cloud.points)}")
    if args.save:
        save_cloud(args.save, cloud)
        print(f"Saved point cloud to {args.save}")

    o3d.visualization.draw_geometries([cloud])


if __name__ == "__main__":
    main()
