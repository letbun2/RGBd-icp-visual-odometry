from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(poses: list[np.ndarray], output_path: str | None = None) -> None:
    xyz = np.array([T[:3, 3] for T in poses])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], linewidth=2)
    ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], c="green", label="start")
    ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], c="red", label="end")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Estimated RGB-D VO trajectory")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
    else:
        plt.show()

