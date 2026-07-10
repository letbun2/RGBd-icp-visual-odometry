import numpy as np


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        diag = np.diag(R)
        axis = int(np.argmax(diag))
        if axis == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif axis == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    return q / np.linalg.norm(q)


def save_tum_trajectory(path: str, timestamps: list[float], poses: list[np.ndarray]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for timestamp, T in zip(timestamps, poses):
            t = T[:3, 3]
            qx, qy, qz, qw = rotation_matrix_to_quaternion(T[:3, :3])
            f.write(
                f"{timestamp:.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )

