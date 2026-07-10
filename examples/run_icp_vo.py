import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgbd_vo.camera import intrinsics_from_name
from rgbd_vo.dataset import load_associations
from rgbd_vo.geometry import save_tum_trajectory
from rgbd_vo.icp_vo import IcpConfig, run_icp_odometry
from rgbd_vo.visualization import plot_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RGB-D ICP visual odometry.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--association", required=True, type=Path)
    parser.add_argument("--camera", default="freiburg1")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--max-correspondence-distance", type=float, default=0.08)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--point-to-point", action="store_true")
    parser.add_argument("--output", default="outputs/estimated_trajectory.txt")
    parser.add_argument("--plot", default="outputs/trajectory.png")
    args = parser.parse_args()

    frames = load_associations(args.dataset, args.association)
    intr = intrinsics_from_name(args.camera)
    config = IcpConfig(
        voxel_size=args.voxel_size,
        max_correspondence_distance=args.max_correspondence_distance,
        stride=args.stride,
        point_to_plane=not args.point_to_point,
    )

    timestamps, poses, steps = run_icp_odometry(
        frames,
        intr,
        config,
        max_frames=args.max_frames,
    )

    for idx, step in enumerate(steps, start=1):
        print(
            f"{idx:04d} fitness={step.fitness:.4f} "
            f"rmse={step.inlier_rmse:.5f} "
            f"t=({poses[idx][0,3]:.3f}, {poses[idx][1,3]:.3f}, {poses[idx][2,3]:.3f})"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_tum_trajectory(str(output), timestamps, poses)
    print(f"Saved trajectory to {output}")

    plot_trajectory(poses, args.plot)
    print(f"Saved trajectory plot to {args.plot}")


if __name__ == "__main__":
    main()
