from __future__ import annotations

from dataclasses import dataclass

from mv_engine.core.types import Matrix3D, Point3D, PointCloud3D, UInt8Array, Vector3D


@dataclass(slots=True)
class Plane3D:
    normal: Vector3D
    center: Point3D


@dataclass(slots=True)
class VolumeMetadata:
    spacing: Vector3D
    origin: Point3D
    direction_matrix: Matrix3D
    shape: tuple[int, int, int]


@dataclass(slots=True)
class AnnulusGeometry:
    plane: Plane3D
    curve: PointCloud3D


@dataclass(slots=True)
class MitralLandmarks:
    a_point: Point3D
    p_point: Point3D
    al_commissure: Point3D
    pm_commissure: Point3D
    left_trigone: Point3D
    right_trigone: Point3D
    tenting_point: Point3D
    commissure_line_dir: Vector3D
    ap_line_dir: Vector3D
    mv_annulus: AnnulusGeometry
    av_annulus: AnnulusGeometry


@dataclass(slots=True)
class AnnulusMetrics:
    ap_diameter: float
    al_pm_diameter: float
    sphericity_index: float
    intertrigonal_distance: float
    commissural_diameter: float
    saddle_shaped_annulus_area_3d: float
    saddle_shaped_annulus_perimeter_3d: float
    d_shaped_annulus_area_2d: float
    d_shaped_annulus_perimeter_2d: float


@dataclass(slots=True)
class ShapeMetrics:
    annulus_height: float
    non_planar_angle_deg: float
    tenting_height: float
    tenting_volume_ml: float
    coaptation_depth: float
    tenting_area_2d: float
    ao_mv_angle_deg: float


@dataclass(slots=True)
class CoaptationMetrics:
    maximum_prolapse_height: float
    maximal_open_coaptation_gap: float
    maximal_open_coaptation_width: float
    total_open_coaptation_area_3d: float


@dataclass(slots=True)
class LeafletMetrics:
    anterior_leaflet_area_3d: float
    posterior_leaflet_area_3d: float
    distal_anterior_leaflet_angle_deg: float
    posterior_leaflet_angle_deg: float
    anterior_leaflet_length_2d: float
    posterior_leaflet_length_2d: float


@dataclass(slots=True)
class MiscMetrics:
    annulus_area_2d: float
    anterior_closure_line_length_2d: float
    anterior_closure_line_length_3d: float
    posterior_closure_line_length_2d: float
    posterior_closure_line_length_3d: float
    c_shaped_annulus: bool


@dataclass(slots=True)
class MitralValveQuantificationResult:
    landmarks: MitralLandmarks
    annulus_metrics: AnnulusMetrics
    shape_metrics: ShapeMetrics
    coaptation_metrics: CoaptationMetrics
    leaflet_metrics: LeafletMetrics
    misc_metrics: MiscMetrics


@dataclass(slots=True)
class PatientValveData:
    mask_anterior: PointCloud3D
    mask_posterior: PointCloud3D
    mask_annulus: PointCloud3D
    mask_aortic_annulus: PointCloud3D
    metadata: VolumeMetadata


@dataclass(slots=True)
class LabelVolume:
    labels: UInt8Array
    metadata: VolumeMetadata
