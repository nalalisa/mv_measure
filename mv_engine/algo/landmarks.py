from __future__ import annotations

import numpy as np

from mv_engine.core.data_models import Plane3D
from mv_engine.core.types import Point3D, PointCloud3D, Vector3D
from mv_engine.math.geometry import normalize_vector


def extract_ap_points(
    fourier_curve: PointCloud3D,
    plane: Plane3D,
    al_pt: Point3D,
    pm_pt: Point3D,
    mask_ant: PointCloud3D,
) -> tuple[Point3D, Point3D, Vector3D]:
    commissure_vector = (pm_pt - al_pt).astype(np.float64)
    commissure_dir = normalize_vector(commissure_vector)

    ap_dir = np.cross(plane.normal, commissure_dir).astype(np.float64)
    ap_dir = normalize_vector(ap_dir)

    anterior_center = np.mean(mask_ant, axis=0, dtype=np.float64)
    vector_to_anterior = anterior_center - plane.center
    if float(np.dot(vector_to_anterior, ap_dir)) < 0.0:
        ap_dir = (-ap_dir).astype(np.float64)

    vectors_from_center = fourier_curve - plane.center
    projections = vectors_from_center @ ap_dir

    a_index = int(np.argmax(projections))
    p_index = int(np.argmin(projections))

    return (
        fourier_curve[a_index].astype(np.float64),
        fourier_curve[p_index].astype(np.float64),
        ap_dir,
    )
