from __future__ import annotations

import numpy as np

from mv_engine.core.data_models import Plane3D
from mv_engine.core.types import FloatArray, Point3D, PointCloud3D, Vector3D


def compute_centroid(points: PointCloud3D) -> Point3D:
    return np.mean(points, axis=0, dtype=np.float64)


def normalize_vector(vector: Vector3D) -> Vector3D:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / norm


def fit_plane_pca(points: PointCloud3D) -> Plane3D:
    if points.shape[0] < 3:
        raise ValueError("At least 3 points are required to fit a plane.")
    center = compute_centroid(points)
    centered = points - center
    covariance = centered.T @ centered / float(points.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, int(np.argmin(eigenvalues))]
    return Plane3D(normal=normalize_vector(normal.astype(np.float64)), center=center.astype(np.float64))


def project_points_to_plane(points: PointCloud3D, plane: Plane3D) -> PointCloud3D:
    offsets = points - plane.center
    signed_distances = offsets @ plane.normal
    return points - signed_distances[:, np.newaxis] * plane.normal


def signed_point_plane_distance(point: Point3D, plane: Plane3D) -> float:
    return float(np.dot(point - plane.center, plane.normal))


def point_plane_distance(point: Point3D, plane: Plane3D) -> float:
    return abs(signed_point_plane_distance(point, plane))


def build_orthonormal_basis(normal: Vector3D) -> FloatArray:
    n = normalize_vector(normal)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(n, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    axis_u = normalize_vector(np.cross(n, helper))
    axis_v = normalize_vector(np.cross(n, axis_u))
    return np.vstack((axis_u, axis_v, n))


def transform_points_to_plane_frame(points: PointCloud3D, plane: Plane3D) -> PointCloud3D:
    basis = build_orthonormal_basis(plane.normal)
    centered = points - plane.center
    return centered @ basis.T
