import cv2
import numpy as np
import open3d as o3d

from rgbd_vo.camera import CameraIntrinsics


def rgbd_to_pointcloud(
    rgb: np.ndarray,
    depth_raw: np.ndarray,
    intr: CameraIntrinsics,
    stride: int = 2,
) -> o3d.geometry.PointCloud:
    depth = intr.depth_to_meters(depth_raw)
    height, width = depth.shape

    us = np.arange(0, width, stride)
    vs = np.arange(0, height, stride)
    grid_u, grid_v = np.meshgrid(us, vs)
    z = depth[grid_v, grid_u]
    valid = z > 0.0

    u = grid_u[valid].astype(np.float32)
    v = grid_v[valid].astype(np.float32)
    z = z[valid]
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy

    points = np.column_stack((x, y, z)).astype(np.float64)
    colors = rgb[grid_v[valid], grid_u[valid]].astype(np.float64) / 255.0

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def preprocess_cloud(
    cloud: o3d.geometry.PointCloud,
    voxel_size: float = 0.03,
    estimate_normals: bool = True,
) -> o3d.geometry.PointCloud:
    if voxel_size > 0.0:
        cloud = cloud.voxel_down_sample(voxel_size)
    if estimate_normals:
        radius = max(voxel_size * 3.0, 0.05)
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )
    return cloud


def draw_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray | None = None,
) -> None:
    source_vis = o3d.geometry.PointCloud(source)
    target_vis = o3d.geometry.PointCloud(target)
    source_vis.paint_uniform_color([1.0, 0.25, 0.15])
    target_vis.paint_uniform_color([0.15, 0.45, 1.0])
    if transformation is not None:
        source_vis.transform(transformation)
    o3d.visualization.draw_geometries([source_vis, target_vis])


def save_cloud(path: str, cloud: o3d.geometry.PointCloud) -> None:
    o3d.io.write_point_cloud(path, cloud)


def resize_rgb_to_depth(rgb: np.ndarray, depth_raw: np.ndarray) -> np.ndarray:
    if rgb.shape[:2] == depth_raw.shape[:2]:
        return rgb
    width = depth_raw.shape[1]
    height = depth_raw.shape[0]
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)

