from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree

from mv_engine.core.data_models import Plane3D
from mv_engine.core.types import PointCloud2D, PointCloud3D, Vector3D
from mv_engine.math.geometry import polyline_length, vector_to_plane_angle_degrees


@dataclass(slots=True)
class LeafletProfileMetrics:
    area_3d: float
    length_2d: float
    angle_deg: float


def _fit_local_pca_basis(points: PointCloud3D) -> tuple[PointCloud2D, np.ndarray]:
    center = np.mean(points, axis=0, dtype=np.float64)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2].astype(np.float64)
    projected = centered @ basis.T
    return projected.astype(np.float64), center


def _filtered_surface_area(points: PointCloud3D, edge_multiplier: float = 3.5) -> float:
    if points.shape[0] < 3:
        return 0.0

    projected, _ = _fit_local_pca_basis(points)
    try:
        delaunay = Delaunay(projected)
    except QhullError:
        return 0.0

    tree = cKDTree(points)
    nearest_distances, _ = tree.query(points, k=2)
    median_nn_distance = float(np.median(nearest_distances[:, 1])) if nearest_distances.shape[1] > 1 else 1.0
    max_edge_length = max(1.0, median_nn_distance * edge_multiplier)

    total_area = 0.0
    for simplex in delaunay.simplices:
        triangle = points[simplex].astype(np.float64)
        edge_lengths = np.array(
            [
                np.linalg.norm(triangle[0] - triangle[1]),
                np.linalg.norm(triangle[1] - triangle[2]),
                np.linalg.norm(triangle[2] - triangle[0]),
            ],
            dtype=np.float64,
        )
        if float(np.max(edge_lengths)) > max_edge_length:
            continue
        area = 0.5 * float(np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])))
        total_area += area

    return total_area


def _slice_leaflet_profile(
    points: PointCloud3D,
    plane: Plane3D,
    ap_line_dir: Vector3D,
    side_sign: float,
    slice_thickness: float,
) -> PointCloud2D:
    commissure_dir = np.cross(plane.normal, ap_line_dir).astype(np.float64)
    commissure_dir /= np.linalg.norm(commissure_dir)

    relative_points = points - plane.center
    section_offset = relative_points @ commissure_dir
    slice_mask = np.abs(section_offset) <= slice_thickness
    sliced_points = points[slice_mask].astype(np.float64)
    if sliced_points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    relative_sliced = sliced_points - plane.center
    x_coords = relative_sliced @ ap_line_dir
    y_coords = relative_sliced @ plane.normal
    side_mask = x_coords * side_sign >= 0.0
    x_side = x_coords[side_mask]
    y_side = y_coords[side_mask]
    if x_side.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    rounded_x = np.round(x_side, 2)
    unique_x = np.unique(rounded_x)
    profile = np.zeros((unique_x.shape[0], 2), dtype=np.float64)
    for index, x_value in enumerate(unique_x):
        bin_mask = rounded_x == x_value
        profile[index, 0] = float(np.mean(x_side[bin_mask]))
        profile[index, 1] = float(np.min(y_side[bin_mask]))

    if side_sign > 0.0:
        order = np.argsort(profile[:, 0])[::-1]
    else:
        order = np.argsort(profile[:, 0])
    return profile[order].astype(np.float64)


def _profile_length(profile: PointCloud2D) -> float:
    return polyline_length(profile, closed=False)


def _anterior_angle(profile: PointCloud2D) -> float:
    if profile.shape[0] < 2:
        return 0.0
    midpoint_index = profile.shape[0] // 2
    distal_vector = np.array(
        [
            profile[-1, 0] - profile[midpoint_index, 0],
            0.0,
            profile[-1, 1] - profile[midpoint_index, 1],
        ],
        dtype=np.float64,
    )
    return vector_to_plane_angle_degrees(distal_vector, np.array([0.0, 0.0, 1.0], dtype=np.float64))


def _posterior_angle(profile: PointCloud2D) -> float:
    if profile.shape[0] < 2:
        return 0.0
    vector = np.array(
        [
            profile[-1, 0] - profile[0, 0],
            0.0,
            profile[-1, 1] - profile[0, 1],
        ],
        dtype=np.float64,
    )
    return vector_to_plane_angle_degrees(vector, np.array([0.0, 0.0, 1.0], dtype=np.float64))


def compute_leaflet_profile_metrics(
    points: PointCloud3D,
    plane: Plane3D,
    ap_line_dir: Vector3D,
    side_sign: float,
    slice_thickness: float,
) -> LeafletProfileMetrics:
    area_3d = _filtered_surface_area(points)
    profile = _slice_leaflet_profile(points, plane, ap_line_dir, side_sign, slice_thickness)
    length_2d = _profile_length(profile)
    if side_sign > 0.0:
        angle_deg = _anterior_angle(profile)
    else:
        angle_deg = _posterior_angle(profile)
    return LeafletProfileMetrics(area_3d=area_3d, length_2d=length_2d, angle_deg=angle_deg)
