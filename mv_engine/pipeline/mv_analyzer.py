from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mv_engine.algo.annulus_metrics import (
    compute_annulus_metrics,
    compute_annulus_shape_metrics,
    compute_misc_annulus_metrics,
    find_trigones,
)
from mv_engine.algo.coaptation import analyze_coaptation
from mv_engine.algo.commissure import find_commissures
from mv_engine.algo.landmarks import extract_ap_points
from mv_engine.algo.leaflet_metrics import compute_leaflet_profile_metrics
from mv_engine.algo.plane_fitter import extract_annulus_plane
from mv_engine.algo.tenting import compute_tenting_metrics
from mv_engine.core.data_models import (
    AnnulusGeometry,
    AnnulusMetrics,
    CoaptationMetrics,
    LeafletMetrics,
    MiscMetrics,
    MitralLandmarks,
    MitralValveQuantificationResult,
    PatientValveData,
    ShapeMetrics,
)
from mv_engine.io.nrrd_reader import load_patient_valve_data_from_nrrd
from mv_engine.math.geometry import normalize_vector, orient_plane_normal_away_from_points


@dataclass(slots=True)
class AnalyzerConfig:
    voxel_size: float = 0.5
    fourier_degree: int = 3
    aortic_fourier_degree: int = 2
    curve_samples: int = 360
    coaptation_distance_threshold: float = 2.0
    coaptation_search_radius: float = 6.0
    ap_slice_thickness: float = 1.0
    closure_bin_size: float = 1.0


class MitralValveAnalyzer:
    def __init__(self, config: dict | AnalyzerConfig):
        if isinstance(config, AnalyzerConfig):
            self.config = config
        else:
            self.config = AnalyzerConfig(
                voxel_size=float(config.get("voxel_size", 0.5)),
                fourier_degree=int(config.get("fourier_degree", 3)),
                aortic_fourier_degree=int(config.get("aortic_fourier_degree", 2)),
                curve_samples=int(config.get("curve_samples", 360)),
                coaptation_distance_threshold=float(config.get("coaptation_distance_threshold", 2.0)),
                coaptation_search_radius=float(config.get("coaptation_search_radius", 6.0)),
                ap_slice_thickness=float(config.get("ap_slice_thickness", 1.0)),
                closure_bin_size=float(config.get("closure_bin_size", 1.0)),
            )

    def run_quantification_from_nrrd(self, file_path: str) -> MitralValveQuantificationResult:
        raw_data = load_patient_valve_data_from_nrrd(file_path)
        return self.run_quantification(raw_data)

    def run_quantification(self, raw_data: PatientValveData) -> MitralValveQuantificationResult:
        combined_leaflets = np.vstack((raw_data.mask_anterior, raw_data.mask_posterior)).astype(np.float64)

        mv_plane, mv_curve = extract_annulus_plane(
            raw_data.mask_annulus,
            degree=self.config.fourier_degree,
            voxel_size=self.config.voxel_size,
            curve_samples=self.config.curve_samples,
        )
        mv_plane = orient_plane_normal_away_from_points(mv_plane, combined_leaflets)

        av_plane, av_curve = extract_annulus_plane(
            raw_data.mask_aortic_annulus,
            degree=self.config.aortic_fourier_degree,
            voxel_size=self.config.voxel_size,
            curve_samples=self.config.curve_samples,
        )

        al_point, pm_point = find_commissures(
            mv_curve,
            raw_data.mask_anterior,
            raw_data.mask_posterior,
        )
        a_point, p_point, ap_line_dir = extract_ap_points(
            mv_curve,
            mv_plane,
            al_point,
            pm_point,
            raw_data.mask_anterior,
        )

        commissure_line_dir = normalize_vector((pm_point - al_point).astype(np.float64))
        ap_line_dir = normalize_vector(ap_line_dir.astype(np.float64))

        left_trigone, right_trigone = find_trigones(
            mv_curve,
            raw_data.mask_aortic_annulus,
            mv_plane,
            ap_line_dir,
        )

        annulus_metrics = compute_annulus_metrics(
            raw_data.mask_annulus,
            mv_curve,
            mv_plane,
            a_point,
            p_point,
            al_point,
            pm_point,
            left_trigone,
            right_trigone,
            ap_line_dir,
        )

        annulus_height, non_planar_angle_deg, ao_mv_angle_deg = compute_annulus_shape_metrics(
            mv_curve,
            mv_plane,
            mv_plane.normal,
            av_plane.normal,
        )

        coaptation_analysis = analyze_coaptation(
            raw_data.mask_anterior,
            raw_data.mask_posterior,
            mv_plane,
            mv_curve,
            ap_line_dir,
            commissure_line_dir,
            contact_threshold=self.config.coaptation_distance_threshold,
            search_radius=self.config.coaptation_search_radius,
            closure_bin_size=self.config.closure_bin_size,
        )

        tenting_metrics = compute_tenting_metrics(
            raw_data.mask_anterior,
            raw_data.mask_posterior,
            mv_plane,
            mv_curve,
            ap_line_dir,
            voxel_size=self.config.voxel_size,
            coaptation_distance_threshold=self.config.coaptation_distance_threshold,
            ap_slice_thickness=self.config.ap_slice_thickness,
        )

        anterior_leaflet_metrics = compute_leaflet_profile_metrics(
            raw_data.mask_anterior,
            mv_plane,
            ap_line_dir,
            side_sign=1.0,
            slice_thickness=self.config.ap_slice_thickness,
        )
        posterior_leaflet_metrics = compute_leaflet_profile_metrics(
            raw_data.mask_posterior,
            mv_plane,
            ap_line_dir,
            side_sign=-1.0,
            slice_thickness=self.config.ap_slice_thickness,
        )

        annulus_area_2d, c_shaped_annulus = compute_misc_annulus_metrics(
            mv_curve,
            mv_plane,
            ap_line_dir,
            left_trigone,
            right_trigone,
        )

        landmarks = MitralLandmarks(
            a_point=a_point.astype(np.float64),
            p_point=p_point.astype(np.float64),
            al_commissure=al_point.astype(np.float64),
            pm_commissure=pm_point.astype(np.float64),
            left_trigone=left_trigone.astype(np.float64),
            right_trigone=right_trigone.astype(np.float64),
            tenting_point=coaptation_analysis.coaptation_point.astype(np.float64),
            commissure_line_dir=commissure_line_dir.astype(np.float64),
            ap_line_dir=ap_line_dir.astype(np.float64),
            mv_annulus=AnnulusGeometry(plane=mv_plane, curve=mv_curve.astype(np.float64)),
            av_annulus=AnnulusGeometry(plane=av_plane, curve=av_curve.astype(np.float64)),
        )

        shape_metrics = ShapeMetrics(
            annulus_height=annulus_height,
            non_planar_angle_deg=non_planar_angle_deg,
            tenting_height=tenting_metrics.height,
            tenting_volume_ml=tenting_metrics.volume,
            coaptation_depth=coaptation_analysis.coaptation_depth,
            tenting_area_2d=tenting_metrics.area,
            ao_mv_angle_deg=ao_mv_angle_deg,
        )

        coaptation_metrics = CoaptationMetrics(
            maximum_prolapse_height=coaptation_analysis.maximum_prolapse_height,
            maximal_open_coaptation_gap=coaptation_analysis.maximal_open_coaptation_gap,
            maximal_open_coaptation_width=coaptation_analysis.maximal_open_coaptation_width,
            total_open_coaptation_area_3d=coaptation_analysis.total_open_coaptation_area_3d,
        )

        leaflet_metrics = LeafletMetrics(
            anterior_leaflet_area_3d=anterior_leaflet_metrics.area_3d,
            posterior_leaflet_area_3d=posterior_leaflet_metrics.area_3d,
            distal_anterior_leaflet_angle_deg=anterior_leaflet_metrics.angle_deg,
            posterior_leaflet_angle_deg=posterior_leaflet_metrics.angle_deg,
            anterior_leaflet_length_2d=anterior_leaflet_metrics.length_2d,
            posterior_leaflet_length_2d=posterior_leaflet_metrics.length_2d,
        )

        misc_metrics = MiscMetrics(
            annulus_area_2d=annulus_area_2d,
            anterior_closure_line_length_2d=coaptation_analysis.anterior_closure_line_length_2d,
            anterior_closure_line_length_3d=coaptation_analysis.anterior_closure_line_length_3d,
            posterior_closure_line_length_2d=coaptation_analysis.posterior_closure_line_length_2d,
            posterior_closure_line_length_3d=coaptation_analysis.posterior_closure_line_length_3d,
            c_shaped_annulus=c_shaped_annulus,
        )

        return MitralValveQuantificationResult(
            landmarks=landmarks,
            annulus_metrics=annulus_metrics,
            shape_metrics=shape_metrics,
            coaptation_metrics=coaptation_metrics,
            leaflet_metrics=leaflet_metrics,
            misc_metrics=misc_metrics,
        )
