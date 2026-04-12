from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from mv_engine.core.types import Point3D, PointCloud3D


def find_commissures(
    fourier_curve: PointCloud3D,
    mask_ant: PointCloud3D,
    mask_post: PointCloud3D,
) -> tuple[Point3D, Point3D]:
    tree_ant = cKDTree(mask_ant)
    tree_post = cKDTree(mask_post)

    distance_ant, _ = tree_ant.query(fourier_curve)
    distance_post, _ = tree_post.query(fourier_curve)
    total_distance = distance_ant + distance_post

    first_index = int(np.argmin(total_distance))

    n_points = int(fourier_curve.shape[0])
    keep_out_zone = n_points // 4
    valid_indices: list[int] = []

    for index in range(n_points):
        circular_distance = min(abs(index - first_index), n_points - abs(index - first_index))
        if circular_distance > keep_out_zone:
            valid_indices.append(index)

    valid_indices_array = np.asarray(valid_indices, dtype=np.int64)
    second_relative_index = int(np.argmin(total_distance[valid_indices_array]))
    second_index = int(valid_indices_array[second_relative_index])

    return (
        fourier_curve[first_index].astype(np.float64),
        fourier_curve[second_index].astype(np.float64),
    )
