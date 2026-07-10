from dataclasses import dataclass

import numpy as np
import open3d as o3d

from rgbd_vo.camera import CameraIntrinsics
from rgbd_vo.dataset import RGBDFrame, read_rgb_depth
from rgbd_vo.pointcloud import preprocess_cloud, resize_rgb_to_depth, rgbd_to_pointcloud


@dataclass
class IcpConfig:
    voxel_size: float = 0.03
    max_correspondence_distance: float = 0.08
    stride: int = 2
    point_to_plane: bool = True


@dataclass
class IcpStepResult:
    transformation: np.ndarray
    fitness: float
    inlier_rmse: float


def estimate_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    config: IcpConfig,
    init: np.ndarray | None = None,
) -> IcpStepResult:
    if init is None:
        init = np.eye(4)

    method: o3d.pipelines.registration.TransformationEstimation
    if config.point_to_plane:
        method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        method = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        config.max_correspondence_distance,
        init,
        method,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
    )

    return IcpStepResult(
        transformation=result.transformation,
        fitness=float(result.fitness),
        inlier_rmse=float(result.inlier_rmse),
    )


def frame_to_cloud(
    frame: RGBDFrame,
    intr: CameraIntrinsics,
    config: IcpConfig,
) -> o3d.geometry.PointCloud:
    rgb, depth_raw = read_rgb_depth(frame)
    rgb = resize_rgb_to_depth(rgb, depth_raw)
    cloud = rgbd_to_pointcloud(rgb, depth_raw, intr, stride=config.stride)
    return preprocess_cloud(cloud, voxel_size=config.voxel_size, estimate_normals=True)


def run_icp_odometry(
    frames: list[RGBDFrame],
    intr: CameraIntrinsics,
    config: IcpConfig,
    max_frames: int | None = None,
) -> tuple[list[float], list[np.ndarray], list[IcpStepResult]]:
    if max_frames is not None:
        frames = frames[:max_frames]
    if len(frames) < 2:
        raise ValueError("Need at least two frames")

    timestamps = [frames[0].timestamp]
    poses = [np.eye(4)]
    step_results: list[IcpStepResult] = []

    prev_cloud = frame_to_cloud(frames[0], intr, config)
    for idx in range(1, len(frames)):
        curr_cloud = frame_to_cloud(frames[idx], intr, config)
        step = estimate_icp(curr_cloud, prev_cloud, config)

        T_world_prev = poses[-1]
        T_prev_curr = step.transformation
        T_world_curr = T_world_prev @ T_prev_curr

        timestamps.append(frames[idx].timestamp)
        poses.append(T_world_curr)
        step_results.append(step)
        prev_cloud = curr_cloud

    return timestamps, poses, step_results
