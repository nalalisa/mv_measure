from __future__ import annotations

from dataclasses import dataclass

from mv_engine.core.types import Point3D, PointCloud3D, Vector3D


@dataclass(slots=True)
class Plane3D:
    normal: Vector3D
    center: Point3D


@dataclass(slots=True)
class MitralLandmarks:
    a_point: Point3D
    p_point: Point3D
    al_commissure: Point3D
    pm_commissure: Point3D
    tenting_point: Point3D
    commissure_line_dir: Vector3D
    ap_line_dir: Vector3D
    annulus_plane: Plane3D
    annulus_curve: PointCloud3D
    tenting_height: float
    tenting_area: float
    tenting_volume: float
    annulus_perimeter: float


@dataclass(slots=True)
class PatientValveData:
    mask_anterior: PointCloud3D
    mask_posterior: PointCloud3D
    mask_annulus: PointCloud3D
