from __future__ import annotations

import numpy as np
import nrrd

from mv_engine.io.nrrd_reader import (
    LABEL_ANTERIOR,
    LABEL_AORTIC_ANNULUS,
    LABEL_MV_ANNULUS,
    LABEL_POSTERIOR,
    build_label_volume_from_numpy,
    build_patient_valve_data_from_label_volume,
)
from mv_engine.pipeline.mv_analyzer import MitralValveAnalyzer


def _stamp_points(volume: np.ndarray, points: np.ndarray, label_value: int) -> None:
    rounded_points = np.rint(points).astype(np.int64)
    for point in rounded_points:
        x_coord = int(np.clip(point[0], 0, volume.shape[0] - 1))
        y_coord = int(np.clip(point[1], 0, volume.shape[1] - 1))
        z_coord = int(np.clip(point[2], 0, volume.shape[2] - 1))
        volume[x_coord, y_coord, z_coord] = label_value


def build_fixture_volume() -> np.ndarray:
    volume = np.zeros((128, 128, 128), dtype=np.uint8)
    center = np.array([64.0, 64.0, 64.0], dtype=np.float64)

    theta = np.linspace(-np.pi, np.pi, 240, endpoint=False, dtype=np.float64)
    mv_annulus = np.column_stack(
        (
            center[0] + 22.0 * np.cos(theta),
            center[1] + 15.0 * np.sin(theta),
            center[2] + 2.0 * np.cos(2.0 * theta),
        )
    ).astype(np.float64)
    aortic_annulus = np.column_stack(
        (
            center[0] + 10.0 + 6.0 * np.cos(theta),
            center[1] + 4.0 * np.sin(theta),
            center[2] + 6.0 + 1.5 * np.cos(theta),
        )
    ).astype(np.float64)

    anterior_points: list[np.ndarray] = []
    posterior_points: list[np.ndarray] = []
    for u_value in np.linspace(0.0, 1.0, 32, dtype=np.float64):
        for v_value in np.linspace(-1.0, 1.0, 32, dtype=np.float64):
            anterior_points.append(
                np.array(
                    [
                        center[0] + 18.0 * (1.0 - u_value) + 0.35 * u_value,
                        center[1] + 10.0 * v_value,
                        center[2] - 7.0 * u_value - 1.2 * (1.0 - v_value * v_value),
                    ],
                    dtype=np.float64,
                )
            )
            posterior_points.append(
                np.array(
                    [
                        center[0] - 16.0 * (1.0 - u_value) - 0.35 * u_value,
                        center[1] + 9.5 * v_value,
                        center[2] - 7.0 * u_value - 1.1 * (1.0 - v_value * v_value),
                    ],
                    dtype=np.float64,
                )
            )

    _stamp_points(volume, mv_annulus, LABEL_MV_ANNULUS)
    _stamp_points(volume, aortic_annulus, LABEL_AORTIC_ANNULUS)
    _stamp_points(volume, np.vstack(anterior_points).astype(np.float64), LABEL_ANTERIOR)
    _stamp_points(volume, np.vstack(posterior_points).astype(np.float64), LABEL_POSTERIOR)
    return volume


def test_build_patient_valve_data_from_label_volume() -> None:
    label_volume = build_label_volume_from_numpy(build_fixture_volume())
    patient_data = build_patient_valve_data_from_label_volume(label_volume)
    assert patient_data.mask_anterior.shape[0] > 0
    assert patient_data.mask_posterior.shape[0] > 0
    assert patient_data.mask_annulus.shape[0] > 0
    assert patient_data.mask_aortic_annulus.shape[0] > 0
    assert patient_data.metadata.shape == (128, 128, 128)


def test_run_quantification_returns_single_frame_measurements() -> None:
    label_volume = build_label_volume_from_numpy(build_fixture_volume())
    patient_data = build_patient_valve_data_from_label_volume(label_volume)

    analyzer = MitralValveAnalyzer(config={"voxel_size": 1.0, "fourier_degree": 3})
    result = analyzer.run_quantification(patient_data)

    assert np.isclose(np.linalg.norm(result.landmarks.mv_annulus.plane.normal), 1.0)
    assert np.isclose(np.linalg.norm(result.landmarks.commissure_line_dir), 1.0)
    assert np.isclose(np.linalg.norm(result.landmarks.ap_line_dir), 1.0)

    assert result.annulus_metrics.ap_diameter > 0.0
    assert result.annulus_metrics.commissural_diameter > 0.0
    assert result.annulus_metrics.saddle_shaped_annulus_area_3d > 0.0
    assert result.annulus_metrics.saddle_shaped_annulus_perimeter_3d > 0.0

    assert result.shape_metrics.annulus_height >= 0.0
    assert result.shape_metrics.tenting_volume_ml >= 0.0
    assert result.shape_metrics.tenting_area_2d >= 0.0

    assert result.coaptation_metrics.maximum_prolapse_height >= 0.0
    assert result.coaptation_metrics.maximal_open_coaptation_gap >= 0.0
    assert result.coaptation_metrics.maximal_open_coaptation_width >= 0.0
    assert result.coaptation_metrics.total_open_coaptation_area_3d >= 0.0

    assert result.leaflet_metrics.anterior_leaflet_area_3d >= 0.0
    assert result.leaflet_metrics.posterior_leaflet_area_3d >= 0.0
    assert result.leaflet_metrics.anterior_leaflet_length_2d >= 0.0
    assert result.leaflet_metrics.posterior_leaflet_length_2d >= 0.0

    assert result.misc_metrics.annulus_area_2d > 0.0
    assert result.misc_metrics.anterior_closure_line_length_2d >= 0.0
    assert result.misc_metrics.posterior_closure_line_length_3d >= 0.0


def test_run_quantification_from_nrrd(tmp_path) -> None:
    volume = build_fixture_volume()
    file_path = tmp_path / "synthetic_case.nrrd"
    nrrd.write(str(file_path), volume, header={"spacings": [1.0, 1.0, 1.0]})

    analyzer = MitralValveAnalyzer(config={"voxel_size": 1.0, "fourier_degree": 3})
    result = analyzer.run_quantification_from_nrrd(str(file_path))

    assert result.annulus_metrics.ap_diameter > 0.0
    assert result.shape_metrics.ao_mv_angle_deg >= 0.0
