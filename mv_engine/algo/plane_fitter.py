from __future__ import annotations

import numpy as np

from mv_engine.algo.preprocessing import downsample_voxel_grid
from mv_engine.core.data_models import Plane3D
from mv_engine.core.types import PointCloud3D, Vector3D
from mv_engine.math.fourier import fit_fourier_series, reconstruct_fourier_curve
from mv_engine.math.geometry import fit_plane_pca, normalize_vector, project_points_to_plane


def _enforce_positive_z_axis(axis: Vector3D) -> Vector3D:
    if axis[2] < 0.0:
        return (-axis).astype(np.float64)
    return axis.astype(np.float64)


def extract_annulus_plane(
    points: PointCloud3D,
    degree: int = 3,
    voxel_size: float = 0.0,
    curve_samples: int = 360,
) -> tuple[Plane3D, PointCloud3D]:
    filtered_points = downsample_voxel_grid(points, voxel_size)
    if filtered_points.shape[0] < 3:
        raise ValueError("At least 3 annulus points are required.")
    if filtered_points.shape[0] < 8:
        plane = fit_plane_pca(filtered_points)
        projected = project_points_to_plane(filtered_points, plane).astype(np.float64)
        return plane, projected

    center = np.mean(filtered_points, axis=0, dtype=np.float64)
    centered_points = filtered_points - center

    covariance_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    eigen_order = np.argsort(eigenvalues)
    ordered_eigenvectors = eigenvectors[:, eigen_order]

    z_axis = _enforce_positive_z_axis(ordered_eigenvectors[:, 0])
    x_axis = normalize_vector(ordered_eigenvectors[:, 1].astype(np.float64))
    y_axis = normalize_vector(ordered_eigenvectors[:, 2].astype(np.float64))
    rotation_matrix = np.vstack((x_axis, y_axis, z_axis)).astype(np.float64)

    rotated_points = centered_points @ rotation_matrix.T
    radius = np.sqrt(rotated_points[:, 0] ** 2 + rotated_points[:, 1] ** 2)
    theta = np.arctan2(rotated_points[:, 1], rotated_points[:, 0]).astype(np.float64)
    z_values = rotated_points[:, 2].astype(np.float64)

    coeffs = fit_fourier_series(theta, z_values, degree)

    a0 = coeffs[0]
    a1 = coeffs[1] if coeffs.shape[0] > 1 else 0.0
    b1 = coeffs[2] if coeffs.shape[0] > 2 else 0.0

    normal_rotated = np.array([-a1, -b1, 1.0], dtype=np.float64)
    normal_rotated = normalize_vector(normal_rotated)

    normal_global = normalize_vector((normal_rotated @ rotation_matrix).astype(np.float64))
    final_center = (center + (z_axis * a0)).astype(np.float64)

    theta_smooth = np.linspace(-np.pi, np.pi, curve_samples, endpoint=False, dtype=np.float64)
    z_smooth = reconstruct_fourier_curve(coeffs, theta_smooth, degree)
    radius_mean = float(np.mean(radius))
    curve_rotated = np.column_stack(
        (
            radius_mean * np.cos(theta_smooth),
            radius_mean * np.sin(theta_smooth),
            z_smooth,
        )
    ).astype(np.float64)
    curve_global = (curve_rotated @ rotation_matrix + center).astype(np.float64)

    return Plane3D(normal=normal_global, center=final_center), curve_global
