from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mv_engine.algo.commissure import find_commissures
from mv_engine.algo.landmarks import extract_ap_points
from mv_engine.algo.measurements import calc_annulus_perimeter
from mv_engine.algo.plane_fitter import extract_annulus_plane
from mv_engine.algo.tenting import compute_tenting_metrics
from mv_engine.core.data_models import MitralLandmarks, PatientValveData
from mv_engine.math.geometry import normalize_vector


@dataclass(slots=True)
class AnalyzerConfig:
    voxel_size: float = 0.0
    fourier_degree: int = 3
    coaptation_distance_threshold: float = 2.0
    ap_slice_thickness: float = 1.0


class MitralValveAnalyzer:
    def __init__(self, config: dict | AnalyzerConfig):
        if isinstance(config, AnalyzerConfig):
            self.config = config
        else:
            self.config = AnalyzerConfig(
                voxel_size=float(config.get("voxel_size", 0.0)),
                fourier_degree=int(config.get("fourier_degree", 3)),
                coaptation_distance_threshold=float(config.get("coaptation_distance_threshold", 2.0)),
                ap_slice_thickness=float(config.get("ap_slice_thickness", 1.0)),
            )

    def run_quantification(self, raw_data: PatientValveData) -> MitralLandmarks:
        annulus_plane, annulus_curve = extract_annulus_plane(
            raw_data.mask_annulus,
            degree=self.config.fourier_degree,
            voxel_size=self.config.voxel_size,
        )
        al_pt, pm_pt = find_commissures(
            annulus_curve,
            raw_data.mask_anterior,
            raw_data.mask_posterior,
        )
        a_point, p_point, commissure_line_dir = extract_ap_points(
            annulus_curve,
            annulus_plane,
            al_pt,
            pm_pt,
            raw_data.mask_anterior,
        )
        true_commissure_line_dir = normalize_vector((pm_pt - al_pt).astype(np.float64))
        ap_line_dir = normalize_vector((a_point - p_point).astype(np.float64))
        tenting_metrics = compute_tenting_metrics(
            raw_data.mask_anterior,
            raw_data.mask_posterior,
            annulus_plane,
            annulus_curve,
            ap_line_dir,
            voxel_size=self.config.voxel_size,
            coaptation_distance_threshold=self.config.coaptation_distance_threshold,
            ap_slice_thickness=self.config.ap_slice_thickness,
        )
        annulus_perimeter = calc_annulus_perimeter(annulus_curve)
        return MitralLandmarks(
            a_point=a_point,
            p_point=p_point,
            al_commissure=al_pt,
            pm_commissure=pm_pt,
            tenting_point=tenting_metrics.point,
            commissure_line_dir=true_commissure_line_dir,
            ap_line_dir=ap_line_dir,
            annulus_plane=annulus_plane,
            annulus_curve=annulus_curve,
            tenting_height=tenting_metrics.height,
            tenting_area=tenting_metrics.area,
            tenting_volume=tenting_metrics.volume,
            annulus_perimeter=annulus_perimeter,
        )
