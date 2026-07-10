# RGB-D Visual Odometry Starter

这是一个适合新手上手 SLAM 的 RGB-D 视觉里程计小项目。它支持 TUM RGB-D 数据集，包含：

- 读取 RGB 图、深度图和时间戳关联文件
- 深度图反投影为彩色点云
- 两帧点云 ICP 配准
- 连续帧位姿累计
- 保存 TUM trajectory 格式
- 绘制估计轨迹

## 1. 安装环境

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

如果 Open3D 在你的平台安装较慢，可以先只安装其他依赖，确认数据读取和图像流程，再安装 Open3D。

## 2. 准备 TUM RGB-D 数据集

下载一个 TUM RGB-D sequence，例如 `freiburg1_xyz`，解压后目录大致如下：

```text
rgbd_dataset_freiburg1_xyz/
  rgb.txt
  depth.txt
  groundtruth.txt
  rgb/
  depth/
```

先生成 RGB 和 depth 的关联文件：

```bash
python examples/associate_tum.py --dataset path/to/rgbd_dataset_freiburg1_xyz
```

会生成：

```text
associations.txt
```

## 3. 单帧深度图转点云

```bash
python examples/view_pointcloud.py \
  --dataset path/to/rgbd_dataset_freiburg1_xyz \
  --association path/to/rgbd_dataset_freiburg1_xyz/associations.txt \
  --index 0
```

这一步用于确认：深度单位、相机内参、RGB-D 对齐都没有明显问题。

## 4. 两帧 ICP 配准

```bash
python examples/run_pair_icp.py \
  --dataset path/to/rgbd_dataset_freiburg1_xyz \
  --association path/to/rgbd_dataset_freiburg1_xyz/associations.txt \
  --source-index 1 \
  --target-index 0
```

含义：估计一个变换，把第 1 帧点云对齐到第 0 帧点云。

输出中最重要的是：

```text
transformation:
[[... 4x4 matrix ...]]
fitness:
inlier_rmse:
```

## 5. 连续 RGB-D VO

```bash
python examples/run_icp_vo.py \
  --dataset path/to/rgbd_dataset_freiburg1_xyz \
  --association path/to/rgbd_dataset_freiburg1_xyz/associations.txt \
  --max-frames 200 \
  --output outputs/estimated_trajectory.txt \
  --plot outputs/trajectory.png
```

输出：

- `estimated_trajectory.txt`: TUM trajectory 格式
- `trajectory.png`: 估计轨迹图

## 6. 坐标系约定

本项目使用：

```text
T_a_b 表示把 b 坐标系下的点变换到 a 坐标系。
```

ICP 中：

```text
T_target_source = estimate(source_cloud, target_cloud)
```

也就是：

```text
p_target ~= T_target_source @ p_source
```

连续里程计中，我们维护每一帧相机到世界坐标系的位姿：

```text
T_world_camera_k
```

如果 ICP 得到：

```text
T_prev_curr
```

则累计：

```text
T_world_curr = T_world_prev @ T_prev_curr
```

## 7. 推荐学习顺序

1. 先运行 `view_pointcloud.py`，确认你能看到单帧彩色点云。
2. 运行 `run_pair_icp.py`，看两帧配准前后效果。
3. 运行 `run_icp_vo.py`，生成一小段轨迹。
4. 修改参数：`voxel-size`、`max-correspondence-distance`、`stride`，观察轨迹变化。
5. 再加入 ground truth 评估、ORB + PnP、ROS2。

## 8. 常见问题

### 点云很奇怪

优先检查 depth scale。TUM RGB-D 常用深度缩放是 `5000`，即：

```text
depth_meters = raw_depth / 5000.0
```

### ICP 结果乱跳

常见原因：

- 相邻帧间隔太大
- 点云太稀疏或太密
- `max-correspondence-distance` 不合适
- 场景缺少几何结构
- 相机快速旋转导致初值太差

### 轨迹越来越漂

这是正常的。纯视觉里程计没有回环检测，误差会累计。
