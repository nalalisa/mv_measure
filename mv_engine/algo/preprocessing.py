from __future__ import annotations

import numpy as np

from mv_engine.core.types import PointCloud3D


def downsample_voxel_grid(points: PointCloud3D, voxel_size: float = 1.0) -> PointCloud3D:
    if voxel_size <= 0.0 or points.shape[0] == 0:
        return points.astype(np.float64, copy=True)

    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    unique_indices.sort()
    return points[unique_indices].astype(np.float64)


def voxel_downsample(points: PointCloud3D, voxel_size: float) -> PointCloud3D:
    return downsample_voxel_grid(points, voxel_size)
