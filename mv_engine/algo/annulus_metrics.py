from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from mv_engine.algo.measurements import calc_annulus_perimeter
from mv_engine.core.data_models import AnnulusMetrics, Plane3D
from mv_engine.core.types import Point3D, PointCloud2D, PointCloud3D, Vector3D
from mv_engine.math.geometry import (
    angle_between_vectors_degrees,
    fan_triangulation_area_3d,
    polygon_area_2d,
    polyline_length,
    project_points_to_plane_basis,
    signed_distances_to_plane,
)


def _circular_arc_indices(start_index: int, end_index: int, n_points: int, forward: bool) -> np.ndarray:
    if forward:
        if start_index <= end_index:
            return np.arange(start_index, end_index + 1, dtype=np.int64)
        return np.concatenate(
            (
                np.arange(start_index, n_points, dtype=np.int64),
                np.arange(0, end_index + 1, dtype=np.int64),
            )
        )

    if end_index <= start_index:
        return np.arange(end_index, start_index + 1, dtype=np.int64)[::-1]
    return np.concatenate(
        (
            np.arange(end_index, n_points, dtype=np.int64)[::-1],
            np.arange(0, start_index + 1, dtype=np.int64)[::-1],
        )
    )


def _find_curve_index(curve: PointCloud3D, point: Point3D) -> int:
    distances = np.linalg.norm(curve - point, axis=1)
    return int(np.argmin(distances))


def _project_curve(
    curve: PointCloud3D,
    plane: Plane3D,
    ap_line_dir: Vector3D,
) -> PointCloud2D:
    return project_points_to_plane_basis(curve, plane.center, ap_line_dir, plane.normal)


def find_trigones(
    mv_annulus_curve: PointCloud3D,
    aortic_annulus_points: PointCloud3D,
    mv_plane: Plane3D,
    ap_line_dir: Vector3D,
) -> tuple[Point3D, Point3D]:
    if aortic_annulus_points.shape[0] == 0:
        anterior_projection = (mv_annulus_curve - mv_plane.center) @ ap_line_dir
        anterior_indices = np.where(anterior_projection > 0.0)[0]
        if anterior_indices.shape[0] < 2:
            return mv_annulus_curve[0].astype(np.float64), mv_annulus_curve[len(mv_annulus_curve) // 2].astype(np.float64)
        left_index = int(anterior_indices[0])
        right_index = int(anterior_indices[-1])
        return mv_annulus_curve[left_index].astype(np.float64), mv_annulus_curve[right_index].astype(np.float64)

    tree = cKDTree(aortic_annulus_points)
    distances, _ = tree.query(mv_annulus_curve)
    anterior_projection = (mv_annulus_curve - mv_plane.center) @ ap_line_dir
    valid_indices = np.where(anterior_projection > 0.0)[0]
    if valid_indices.shape[0] == 0:
        valid_indices = np.arange(mv_annulus_curve.shape[0], dtype=np.int64)

    masked_distances = np.full(mv_annulus_curve.shape[0], np.inf, dtype=np.float64)
    masked_distances[valid_indices] = distances[valid_indices]

    first_index = int(np.argmin(masked_distances))
    keep_out = max(6, mv_annulus_curve.shape[0] // 8)

    candidate_indices: list[int] = []
    for index in valid_indices.tolist():
        circular_distance = min(abs(index - first_index), mv_annulus_curve.shape[0] - abs(index - first_index))
        if circular_distance > keep_out:
            candidate_indices.append(index)

    if len(candidate_indices) == 0:
        second_index = (first_index + mv_annulus_curve.shape[0] // 4) % mv_annulus_curve.shape[0]
    else:
        candidate_indices_array = np.asarray(candidate_indices, dtype=np.int64)
        second_index = int(candidate_indices_array[np.argmin(masked_distances[candidate_indices_array])])

    first_point = mv_annulus_curve[first_index].astype(np.float64)
    second_point = mv_annulus_curve[second_index].astype(np.float64)

    commissure_axis = np.cross(mv_plane.normal, ap_line_dir).astype(np.float64)
    first_projection = float(np.dot(first_point - mv_plane.center, commissure_axis))
    second_projection = float(np.dot(second_point - mv_plane.center, commissure_axis))
    if first_projection <= second_projection:
        return first_point, second_point
    return second_point, first_point


def _extract_posterior_arc(
    annulus_curve: PointCloud3D,
    plane: Plane3D,
    ap_line_dir: Vector3D,
    left_trigone: Point3D,
    right_trigone: Point3D,
) -> PointCloud3D:
    left_index = _find_curve_index(annulus_curve, left_trigone)
    right_index = _find_curve_index(annulus_curve, right_trigone)
    curve_projected = _project_curve(annulus_curve, plane, ap_line_dir)

    forward_indices = _circular_arc_indices(left_index, right_index, annulus_curve.shape[0], forward=True)
    backward_indices = _circular_arc_indices(left_index, right_index, annulus_curve.shape[0], forward=False)

    forward_projection_mean = float(np.mean(curve_projected[forward_indices, 0]))
    backward_projection_mean = float(np.mean(curve_projected[backward_indices, 0]))

    if forward_projection_mean < backward_projection_mean:
        return annulus_curve[forward_indices].astype(np.float64)
    return annulus_curve[backward_indices].astype(np.float64)


def _build_d_shaped_polygon_2d(
    annulus_curve: PointCloud3D,
    plane: Plane3D,
    ap_line_dir: Vector3D,
    left_trigone: Point3D,
    right_trigone: Point3D,
) -> PointCloud2D:
    posterior_arc = _extract_posterior_arc(annulus_curve, plane, ap_line_dir, left_trigone, right_trigone)
    projected_arc = _project_curve(posterior_arc, plane, ap_line_dir)
    return projected_arc.astype(np.float64)


def compute_annulus_metrics(
    annulus_points: PointCloud3D,
    annulus_curve: PointCloud3D,
    plane: Plane3D,
    a_point: Point3D,
    p_point: Point3D,
    al_point: Point3D,
    pm_point: Point3D,
    left_trigone: Point3D,
    right_trigone: Point3D,
    ap_line_dir: Vector3D,
) -> AnnulusMetrics:
    projected_curve = _project_curve(annulus_curve, plane, ap_line_dir)

    a_projected = _project_curve(a_point.reshape(1, 3), plane, ap_line_dir)[0]
    p_projected = _project_curve(p_point.reshape(1, 3), plane, ap_line_dir)[0]
    al_projected = _project_curve(al_point.reshape(1, 3), plane, ap_line_dir)[0]
    pm_projected = _project_curve(pm_point.reshape(1, 3), plane, ap_line_dir)[0]

    ap_diameter = float(np.linalg.norm(a_projected - p_projected))
    al_pm_diameter = float(np.linalg.norm(pm_point - al_point))
    commissural_diameter = float(np.linalg.norm(pm_projected - al_projected))
    sphericity_index = ap_diameter / commissural_diameter if commissural_diameter > 0.0 else 0.0
    intertrigonal_distance = float(np.linalg.norm(right_trigone - left_trigone))

    saddle_area_3d = fan_triangulation_area_3d(annulus_curve, plane.center)
    saddle_perimeter_3d = calc_annulus_perimeter(annulus_curve)

    d_shaped_polygon = _build_d_shaped_polygon_2d(
        annulus_curve,
        plane,
        ap_line_dir,
        left_trigone,
        right_trigone,
    )
    d_shaped_area_2d = polygon_area_2d(d_shaped_polygon)
    d_shaped_perimeter_2d = polyline_length(d_shaped_polygon, closed=False) + float(
        np.linalg.norm(d_shaped_polygon[0] - d_shaped_polygon[-1])
    )
    _ = annulus_points

    return AnnulusMetrics(
        ap_diameter=ap_diameter,
        al_pm_diameter=al_pm_diameter,
        sphericity_index=sphericity_index,
        intertrigonal_distance=intertrigonal_distance,
        commissural_diameter=commissural_diameter,
        saddle_shaped_annulus_area_3d=saddle_area_3d,
        saddle_shaped_annulus_perimeter_3d=saddle_perimeter_3d,
        d_shaped_annulus_area_2d=d_shaped_area_2d,
        d_shaped_annulus_perimeter_2d=d_shaped_perimeter_2d,
    )


def compute_annulus_shape_metrics(
    annulus_curve: PointCloud3D,
    plane: Plane3D,
    mv_normal: Vector3D,
    av_normal: Vector3D,
) -> tuple[float, float, float]:
    signed_heights = signed_distances_to_plane(annulus_curve, plane)
    annulus_height = float(np.max(signed_heights) - np.min(signed_heights))

    highest_point = annulus_curve[int(np.argmax(signed_heights))]
    lowest_point = annulus_curve[int(np.argmin(signed_heights))]
    non_planar_angle_deg = angle_between_vectors_degrees(highest_point - plane.center, lowest_point - plane.center)

    ao_mv_angle_deg = angle_between_vectors_degrees(mv_normal, av_normal)
    if ao_mv_angle_deg > 90.0:
        ao_mv_angle_deg = 180.0 - ao_mv_angle_deg

    return annulus_height, non_planar_angle_deg, ao_mv_angle_deg


def compute_misc_annulus_metrics(
    annulus_curve: PointCloud3D,
    plane: Plane3D,
    ap_line_dir: Vector3D,
    left_trigone: Point3D,
    right_trigone: Point3D,
) -> tuple[float, bool]:
    projected_curve = _project_curve(annulus_curve, plane, ap_line_dir)
    annulus_area_2d = polygon_area_2d(projected_curve)

    projected_left = _project_curve(left_trigone.reshape(1, 3), plane, ap_line_dir)[0]
    projected_right = _project_curve(right_trigone.reshape(1, 3), plane, ap_line_dir)[0]
    trigone_chord = float(np.linalg.norm(projected_right - projected_left))

    posterior_arc = _extract_posterior_arc(annulus_curve, plane, ap_line_dir, left_trigone, right_trigone)
    projected_posterior_arc = _project_curve(posterior_arc, plane, ap_line_dir)
    posterior_arc_length = polyline_length(projected_posterior_arc, closed=False)
    full_perimeter_2d = polyline_length(projected_curve, closed=True)
    anterior_arc_length = max(0.0, full_perimeter_2d - posterior_arc_length)
    c_shaped_annulus = anterior_arc_length > trigone_chord * 1.1
    return annulus_area_2d, c_shaped_annulus
