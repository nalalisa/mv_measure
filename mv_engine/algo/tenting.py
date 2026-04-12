from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from mv_engine.core.data_models import Plane3D
from mv_engine.core.types import Point3D, PointCloud3D, Vector3D
from mv_engine.math.geometry import normalize_vector


@dataclass(slots=True)
class TentingMetrics:
    height: float
    area: float
    volume: float
    point: Point3D


def _extract_coaptation_zone(
    mask_ant: PointCloud3D,
    mask_post: PointCloud3D,
    distance_threshold: float,
) -> PointCloud3D:
    tree_post = cKDTree(mask_post)
    distances, _ = tree_post.query(mask_ant)
    coaptation_points = mask_ant[distances < distance_threshold]
    return coaptation_points.astype(np.float64)


def _compute_tenting_height(
    coaptation_points: PointCloud3D,
    plane: Plane3D,
) -> tuple[float, Point3D]:
    if coaptation_points.shape[0] == 0:
        return 0.0, plane.center.astype(np.float64)

    relative_points = coaptation_points - plane.center
    signed_distances = relative_points @ plane.normal
    absolute_distances = np.abs(signed_distances)
    point_index = int(np.argmax(absolute_distances))
    return float(absolute_distances[point_index]), coaptation_points[point_index].astype(np.float64)


def _compute_tenting_volume(
    mask_ant: PointCloud3D,
    mask_post: PointCloud3D,
    plane: Plane3D,
    voxel_size: float,
) -> float:
    all_leaflet_points = np.vstack((mask_ant, mask_post)).astype(np.float64)
    relative_points = all_leaflet_points - plane.center
    signed_depths = relative_points @ plane.normal
    ventricle_side_depths = signed_depths[signed_depths < 0.0]
    if ventricle_side_depths.shape[0] == 0:
        return 0.0

    column_area = voxel_size * voxel_size
    volume_mm3 = float(np.sum(np.abs(ventricle_side_depths)) * column_area)
    return volume_mm3 / 1000.0


def _build_section_basis(
    plane_normal: Vector3D,
    ap_line_dir: Vector3D,
) -> tuple[Vector3D, Vector3D]:
    axis_x = normalize_vector(ap_line_dir.astype(np.float64))
    axis_y = normalize_vector(plane_normal.astype(np.float64))
    return axis_x, axis_y


def _slice_points_near_ap_plane(
    points: PointCloud3D,
    plane_center: Point3D,
    section_normal: Vector3D,
    slice_thickness: float,
) -> PointCloud3D:
    relative_points = points - plane_center
    signed_offsets = relative_points @ section_normal
    selected_mask = np.abs(signed_offsets) <= slice_thickness
    return points[selected_mask].astype(np.float64)


def _project_to_section_frame(
    points: PointCloud3D,
    origin: Point3D,
    axis_x: Vector3D,
    axis_y: Vector3D,
) -> PointCloud3D:
    relative_points = points - origin
    x_coords = relative_points @ axis_x
    y_coords = relative_points @ axis_y
    return np.column_stack((x_coords, y_coords)).astype(np.float64)


def _select_extreme_curve_points(
    projected_curve: PointCloud3D,
) -> tuple[int, int]:
    positive_index = int(np.argmax(projected_curve[:, 0]))
    negative_index = int(np.argmin(projected_curve[:, 0]))
    return positive_index, negative_index


def _extract_leaflet_profile(
    projected_points: PointCloud3D,
    projected_curve: PointCloud3D,
) -> PointCloud3D:
    if projected_points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    positive_index, negative_index = _select_extreme_curve_points(projected_curve)
    x_min = float(min(projected_curve[positive_index, 0], projected_curve[negative_index, 0]))
    x_max = float(max(projected_curve[positive_index, 0], projected_curve[negative_index, 0]))

    in_range_mask = (projected_points[:, 0] >= x_min) & (projected_points[:, 0] <= x_max)
    profile_points = projected_points[in_range_mask]
    if profile_points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    sort_indices = np.argsort(profile_points[:, 0])
    return profile_points[sort_indices].astype(np.float64)


def _collapse_profile_to_lower_envelope(profile_points: PointCloud3D) -> PointCloud3D:
    if profile_points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    rounded_x = np.round(profile_points[:, 0], decimals=3)
    unique_x_values = np.unique(rounded_x)
    envelope = np.zeros((unique_x_values.shape[0], 2), dtype=np.float64)

    for index, x_value in enumerate(unique_x_values):
        same_x_mask = rounded_x == x_value
        bucket = profile_points[same_x_mask]
        y_min = float(np.min(bucket[:, 1]))
        envelope[index, 0] = float(np.mean(bucket[:, 0]))
        envelope[index, 1] = y_min

    return envelope


def _compute_tenting_area(
    mask_ant: PointCloud3D,
    mask_post: PointCloud3D,
    annulus_curve: PointCloud3D,
    plane: Plane3D,
    ap_line_dir: Vector3D,
    slice_thickness: float,
) -> float:
    commissure_section_normal = np.cross(plane.normal, ap_line_dir).astype(np.float64)
    commissure_section_normal = normalize_vector(commissure_section_normal)

    sliced_ant = _slice_points_near_ap_plane(mask_ant, plane.center, commissure_section_normal, slice_thickness)
    sliced_post = _slice_points_near_ap_plane(mask_post, plane.center, commissure_section_normal, slice_thickness)
    sliced_leaflets = np.vstack((sliced_ant, sliced_post)).astype(np.float64)
    if sliced_leaflets.shape[0] == 0:
        return 0.0

    axis_x, axis_y = _build_section_basis(plane.normal, ap_line_dir)
    projected_curve = _project_to_section_frame(annulus_curve, plane.center, axis_x, axis_y)
    projected_leaflets = _project_to_section_frame(sliced_leaflets, plane.center, axis_x, axis_y)
    profile_points = _extract_leaflet_profile(projected_leaflets, projected_curve)
    envelope = _collapse_profile_to_lower_envelope(profile_points)

    if envelope.shape[0] < 2:
        return 0.0

    envelope_x = envelope[:, 0]
    envelope_y = envelope[:, 1]
    clipped_depth = np.maximum(0.0, -envelope_y)
    return float(np.trapezoid(clipped_depth, envelope_x))


def compute_tenting_metrics(
    mask_ant: PointCloud3D,
    mask_post: PointCloud3D,
    plane: Plane3D,
    annulus_curve: PointCloud3D,
    ap_line_dir: Vector3D,
    voxel_size: float = 0.5,
    coaptation_distance_threshold: float = 2.0,
    ap_slice_thickness: float = 1.0,
) -> TentingMetrics:
    coaptation_points = _extract_coaptation_zone(
        mask_ant,
        mask_post,
        distance_threshold=coaptation_distance_threshold,
    )
    tenting_height, tenting_point = _compute_tenting_height(coaptation_points, plane)
    tenting_area = _compute_tenting_area(
        mask_ant,
        mask_post,
        annulus_curve,
        plane,
        ap_line_dir,
        slice_thickness=ap_slice_thickness,
    )
    effective_voxel_size = voxel_size if voxel_size > 0.0 else 0.5
    tenting_volume = _compute_tenting_volume(
        mask_ant,
        mask_post,
        plane,
        voxel_size=effective_voxel_size,
    )

    return TentingMetrics(
        height=tenting_height,
        area=tenting_area,
        volume=tenting_volume,
        point=tenting_point,
    )
