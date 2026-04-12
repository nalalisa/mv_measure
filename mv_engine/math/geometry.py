from __future__ import annotations

import numpy as np

from mv_engine.core.data_models import Plane3D
from mv_engine.core.types import FloatArray, Matrix3D, Point3D, PointCloud2D, PointCloud3D, Vector3D


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


def signed_distances_to_plane(points: PointCloud3D, plane: Plane3D) -> FloatArray:
    return ((points - plane.center) @ plane.normal).astype(np.float64)


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


def build_plane_basis_from_axis(plane_normal: Vector3D, axis_x: Vector3D) -> Matrix3D:
    axis_x_projected = axis_x - float(np.dot(axis_x, plane_normal)) * plane_normal
    axis_u = normalize_vector(axis_x_projected.astype(np.float64))
    axis_v = normalize_vector(np.cross(plane_normal, axis_u).astype(np.float64))
    return np.vstack((axis_u, axis_v, normalize_vector(plane_normal.astype(np.float64)))).astype(np.float64)


def project_points_to_plane_basis(
    points: PointCloud3D,
    origin: Point3D,
    axis_x: Vector3D,
    plane_normal: Vector3D,
) -> PointCloud2D:
    basis = build_plane_basis_from_axis(plane_normal, axis_x)
    centered = points - origin
    projected = centered @ basis[:2].T
    return projected.astype(np.float64)


def orient_plane_normal_away_from_points(plane: Plane3D, points: PointCloud3D) -> Plane3D:
    if points.shape[0] == 0:
        return plane
    mean_signed_distance = float(np.mean(signed_distances_to_plane(points, plane)))
    if mean_signed_distance > 0.0:
        return Plane3D(normal=(-plane.normal).astype(np.float64), center=plane.center.astype(np.float64))
    return plane


def angle_between_vectors_degrees(vector_a: Vector3D, vector_b: Vector3D) -> float:
    norm_a = normalize_vector(vector_a.astype(np.float64))
    norm_b = normalize_vector(vector_b.astype(np.float64))
    cosine_value = float(np.clip(np.dot(norm_a, norm_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine_value)))


def vector_to_plane_angle_degrees(vector: Vector3D, plane_normal: Vector3D) -> float:
    projected_length = float(np.linalg.norm(vector - np.dot(vector, plane_normal) * plane_normal))
    normal_component = abs(float(np.dot(vector, plane_normal)))
    return float(np.degrees(np.arctan2(normal_component, projected_length)))


def polyline_length(points: PointCloud3D, closed: bool = False) -> float:
    if points.shape[0] < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    if closed:
        closing = (points[0] - points[-1]).reshape(1, -1)
        diffs = np.vstack((diffs, closing))
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def polygon_area_2d(points: PointCloud2D) -> float:
    if points.shape[0] < 3:
        return 0.0
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    return float(0.5 * abs(np.dot(x_coords, np.roll(y_coords, -1)) - np.dot(y_coords, np.roll(x_coords, -1))))


def fan_triangulation_area_3d(curve: PointCloud3D, center: Point3D) -> float:
    if curve.shape[0] < 3:
        return 0.0
    total_area = 0.0
    for index in range(curve.shape[0]):
        point_a = curve[index] - center
        point_b = curve[(index + 1) % curve.shape[0]] - center
        total_area += 0.5 * float(np.linalg.norm(np.cross(point_a, point_b)))
    return total_area
