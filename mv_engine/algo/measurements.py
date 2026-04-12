from __future__ import annotations

import numpy as np

from mv_engine.core.types import PointCloud3D


def calc_annulus_perimeter(fourier_curve: PointCloud3D) -> float:
    diffs = np.diff(fourier_curve, axis=0)
    closing_diff = (fourier_curve[0] - fourier_curve[-1]).reshape(1, 3)
    closed_diffs = np.vstack((diffs, closing_diff))
    distances = np.linalg.norm(closed_diffs, axis=1)
    return float(np.sum(distances))
