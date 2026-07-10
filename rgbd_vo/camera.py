from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 5000.0
    depth_trunc: float = 4.0

    @classmethod
    def tum_freiburg1(cls) -> "CameraIntrinsics":
        return cls(fx=517.3, fy=516.5, cx=318.6, cy=255.3)

    @classmethod
    def tum_freiburg2(cls) -> "CameraIntrinsics":
        return cls(fx=520.9, fy=521.0, cx=325.1, cy=249.7)

    @classmethod
    def tum_freiburg3(cls) -> "CameraIntrinsics":
        return cls(fx=535.4, fy=539.2, cx=320.1, cy=247.6)

    def depth_to_meters(self, depth_raw: np.ndarray) -> np.ndarray:
        depth = depth_raw.astype(np.float32) / float(self.depth_scale)
        depth[depth > self.depth_trunc] = 0.0
        return depth


def intrinsics_from_name(name: str) -> CameraIntrinsics:
    key = name.lower()
    if "freiburg2" in key or key in {"fr2", "tum2", "2"}:
        return CameraIntrinsics.tum_freiburg2()
    if "freiburg3" in key or key in {"fr3", "tum3", "3"}:
        return CameraIntrinsics.tum_freiburg3()
    return CameraIntrinsics.tum_freiburg1()

