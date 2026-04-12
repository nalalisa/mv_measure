from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from mv_engine.core.data_models import Plane3D
from mv_engine.core.types import FloatArray, Point3D, PointCloud2D, PointCloud3D, Vector3D
from mv_engine.math.geometry import (
    polyline_length,
    project_points_to_plane_basis,
    signed_distances_to_plane,
)


@dataclass(slots=True)
class CoaptationAnalysis:
    contact_anterior_line: PointCloud3D
    contact_posterior_line: PointCloud3D
    open_midline: PointCloud3D
    open_gap_distances: FloatArray
    coaptation_point: Point3D
    coaptation_depth: float
    maximum_prolapse_height: float
    maximal_open_coaptation_gap: float
    maximal_open_coaptation_width: float
    total_open_coaptation_area_3d: float
    anterior_closure_line_length_2d: float
    anterior_closure_line_length_3d: float
    posterior_closure_line_length_2d: float
    posterior_closure_line_length_3d: float


def _project_scalar(points: PointCloud3D, origin: Point3D, axis: Vector3D) -> FloatArray:
    return ((points - origin) @ axis).astype(np.float64)


def _extract_central_band(
    points: PointCloud3D,
    plane_center: Point3D,
    ap_line_dir: Vector3D,
    annulus_curve: PointCloud3D,
) -> np.ndarray:
    annulus_projection = np.abs(_project_scalar(annulus_curve, plane_center, ap_line_dir))
    max_extent = float(np.max(annulus_projection)) if annulus_projection.shape[0] > 0 else 0.0
    band_half_width = max(1.0, 0.35 * max_extent)
    point_projection = np.abs(_project_scalar(points, plane_center, ap_line_dir))
    return point_projection <= band_half_width


def _resample_pairs_by_commissure(
    anterior_points: PointCloud3D,
    posterior_points: PointCloud3D,
    distances: FloatArray,
    plane_center: Point3D,
    commissure_line_dir: Vector3D,
    bin_size: float,
    take_min_distance: bool,
) -> tuple[PointCloud3D, PointCloud3D, FloatArray]:
    if anterior_points.shape[0] == 0:
        empty_points = np.empty((0, 3), dtype=np.float64)
        empty_distances = np.empty((0,), dtype=np.float64)
        return empty_points, empty_points, empty_distances

    projections = _project_scalar(anterior_points, plane_center, commissure_line_dir)
    min_projection = float(np.min(projections))
    bin_indices = np.floor((projections - min_projection) / bin_size).astype(np.int64)
    unique_bins = np.unique(bin_indices)

    selected_ant: list[np.ndarray] = []
    selected_post: list[np.ndarray] = []
    selected_distances: list[float] = []

    for bin_value in unique_bins.tolist():
        in_bin = np.where(bin_indices == bin_value)[0]
        if in_bin.shape[0] == 0:
            continue
        if take_min_distance:
            local_index = int(in_bin[np.argmin(distances[in_bin])])
        else:
            local_index = int(in_bin[np.argmax(distances[in_bin])])
        selected_ant.append(anterior_points[local_index])
        selected_post.append(posterior_points[local_index])
        selected_distances.append(float(distances[local_index]))

    ant_array = np.vstack(selected_ant).astype(np.float64) if selected_ant else np.empty((0, 3), dtype=np.float64)
    post_array = np.vstack(selected_post).astype(np.float64) if selected_post else np.empty((0, 3), dtype=np.float64)
    distance_array = np.asarray(selected_distances, dtype=np.float64)

    if ant_array.shape[0] == 0:
        return ant_array, post_array, distance_array

    order = np.argsort(_project_scalar(ant_array, plane_center, commissure_line_dir))
    return ant_array[order], post_array[order], distance_array[order]


def _compute_gap_area_3d(
    anterior_line: PointCloud3D,
    posterior_line: PointCloud3D,
    gap_distances: FloatArray,
) -> float:
    if anterior_line.shape[0] < 2:
        return 0.0
    midline = 0.5 * (anterior_line + posterior_line)
    segment_lengths = np.linalg.norm(np.diff(midline, axis=0), axis=1)
    if segment_lengths.shape[0] == 0:
        return 0.0
    average_gap = 0.5 * (gap_distances[:-1] + gap_distances[1:])
    return float(np.sum(segment_lengths * average_gap))


def _closure_line_lengths(
    line_points: PointCloud3D,
    plane: Plane3D,
    ap_line_dir: Vector3D,
) -> tuple[float, float]:
    length_3d = polyline_length(line_points, closed=False)
    projected_2d = project_points_to_plane_basis(line_points, plane.center, ap_line_dir, plane.normal)
    length_2d = polyline_length(projected_2d, closed=False)
    return length_2d, length_3d


def analyze_coaptation(
    mask_ant: PointCloud3D,
    mask_post: PointCloud3D,
    plane: Plane3D,
    annulus_curve: PointCloud3D,
    ap_line_dir: Vector3D,
    commissure_line_dir: Vector3D,
    contact_threshold: float = 2.0,
    search_radius: float = 6.0,
    closure_bin_size: float = 1.0,
) -> CoaptationAnalysis:
    tree_post = cKDTree(mask_post)
    distances, indices = tree_post.query(mask_ant)
    nearest_post = mask_post[indices].astype(np.float64)

    central_band_mask = _extract_central_band(mask_ant, plane.center, ap_line_dir, annulus_curve)
    candidate_mask = central_band_mask & (distances <= search_radius)

    candidate_ant = mask_ant[candidate_mask].astype(np.float64)
    candidate_post = nearest_post[candidate_mask].astype(np.float64)
    candidate_distances = distances[candidate_mask].astype(np.float64)

    contact_mask = candidate_distances <= contact_threshold
    open_mask = candidate_distances > contact_threshold

    contact_ant, contact_post, contact_distances = _resample_pairs_by_commissure(
        candidate_ant[contact_mask],
        candidate_post[contact_mask],
        candidate_distances[contact_mask],
        plane.center,
        commissure_line_dir,
        bin_size=closure_bin_size,
        take_min_distance=True,
    )

    open_ant, open_post, open_distances = _resample_pairs_by_commissure(
        candidate_ant[open_mask],
        candidate_post[open_mask],
        candidate_distances[open_mask],
        plane.center,
        commissure_line_dir,
        bin_size=closure_bin_size,
        take_min_distance=False,
    )

    if contact_ant.shape[0] > 0:
        coaptation_midline = 0.5 * (contact_ant + contact_post)
        coaptation_signed_depths = signed_distances_to_plane(coaptation_midline, plane)
        deepest_index = int(np.argmin(coaptation_signed_depths))
        coaptation_point = coaptation_midline[deepest_index].astype(np.float64)
        coaptation_depth = float(abs(coaptation_signed_depths[deepest_index]))
    else:
        coaptation_point = plane.center.astype(np.float64)
        coaptation_depth = 0.0

    combined_leaflets = np.vstack((mask_ant, mask_post)).astype(np.float64)
    leaflet_signed_distances = signed_distances_to_plane(combined_leaflets, plane)
    atrial_side_distances = leaflet_signed_distances[leaflet_signed_distances > 0.0]
    maximum_prolapse_height = float(np.max(atrial_side_distances)) if atrial_side_distances.shape[0] > 0 else 0.0

    maximal_open_coaptation_gap = float(np.max(open_distances)) if open_distances.shape[0] > 0 else 0.0
    if open_ant.shape[0] > 0:
        open_proj = _project_scalar(open_ant, plane.center, commissure_line_dir)
        maximal_open_coaptation_width = float(np.max(open_proj) - np.min(open_proj))
    else:
        maximal_open_coaptation_width = 0.0

    total_open_coaptation_area_3d = _compute_gap_area_3d(open_ant, open_post, open_distances)

    anterior_closure_line_length_2d, anterior_closure_line_length_3d = _closure_line_lengths(contact_ant, plane, ap_line_dir)
    posterior_closure_line_length_2d, posterior_closure_line_length_3d = _closure_line_lengths(contact_post, plane, ap_line_dir)

    return CoaptationAnalysis(
        contact_anterior_line=contact_ant,
        contact_posterior_line=contact_post,
        open_midline=(0.5 * (open_ant + open_post)).astype(np.float64) if open_ant.shape[0] > 0 else np.empty((0, 3), dtype=np.float64),
        open_gap_distances=open_distances,
        coaptation_point=coaptation_point,
        coaptation_depth=coaptation_depth,
        maximum_prolapse_height=maximum_prolapse_height,
        maximal_open_coaptation_gap=maximal_open_coaptation_gap,
        maximal_open_coaptation_width=maximal_open_coaptation_width,
        total_open_coaptation_area_3d=total_open_coaptation_area_3d,
        anterior_closure_line_length_2d=anterior_closure_line_length_2d,
        anterior_closure_line_length_3d=anterior_closure_line_length_3d,
        posterior_closure_line_length_2d=posterior_closure_line_length_2d,
        posterior_closure_line_length_3d=posterior_closure_line_length_3d,
    )
