import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgbd_vo.camera import intrinsics_from_name
from rgbd_vo.dataset import load_associations
from rgbd_vo.icp_vo import IcpConfig, estimate_icp, frame_to_cloud
from rgbd_vo.pointcloud import draw_registration


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ICP between two RGB-D frames.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--association", required=True, type=Path)
    parser.add_argument("--source-index", type=int, default=1)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--camera", default="freiburg1")
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--max-correspondence-distance", type=float, default=0.08)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--point-to-point", action="store_true")
    parser.add_argument("--no-view", action="store_true")
    args = parser.parse_args()

    frames = load_associations(args.dataset, args.association)
    intr = intrinsics_from_name(args.camera)
    config = IcpConfig(
        voxel_size=args.voxel_size,
        max_correspondence_distance=args.max_correspondence_distance,
        stride=args.stride,
        point_to_plane=not args.point_to_point,
    )

    source = frame_to_cloud(frames[args.source_index], intr, config)
    target = frame_to_cloud(frames[args.target_index], intr, config)
    result = estimate_icp(source, target, config)

    print("transformation:")
    print(result.transformation)
    print(f"fitness: {result.fitness:.6f}")
    print(f"inlier_rmse: {result.inlier_rmse:.6f}")

    if not args.no_view:
        draw_registration(source, target, result.transformation)


if __name__ == "__main__":
    main()
