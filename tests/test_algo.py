from __future__ import annotations

import numpy as np

from mv_engine.core.data_models import PatientValveData
from mv_engine.pipeline.mv_analyzer import MitralValveAnalyzer


def build_fixture() -> PatientValveData:
    theta = np.linspace(-np.pi, np.pi, 72, endpoint=False, dtype=np.float64)
    radius = 5.0
    annulus = np.column_stack(
        (
            radius * np.cos(theta),
            radius * np.sin(theta),
            0.20 * np.cos(theta) - 0.10 * np.sin(theta),
        )
    ).astype(np.float64)
    anterior = np.array(
        [
            [3.5, 0.0, 0.5],
            [4.0, 0.2, 0.6],
            [2.5, 0.1, 0.4],
        ],
        dtype=np.float64,
    )
    posterior = np.array(
        [
            [-3.6, 0.0, -0.4],
            [-4.2, -0.1, -0.5],
            [-2.8, 0.1, -0.3],
        ],
        dtype=np.float64,
    )
    return PatientValveData(
        mask_anterior=anterior,
        mask_posterior=posterior,
        mask_annulus=annulus,
    )


def test_run_quantification_returns_expected_tips() -> None:
    analyzer = MitralValveAnalyzer(config={"voxel_size": 0.0, "fourier_degree": 2})
    result = analyzer.run_quantification(build_fixture())
    assert np.isclose(np.linalg.norm(result.annulus_plane.normal), 1.0)
    assert np.isclose(np.linalg.norm(result.commissure_line_dir), 1.0)
    assert np.isclose(np.linalg.norm(result.ap_line_dir), 1.0)
    assert result.annulus_curve.shape == (360, 3)
    assert result.tenting_height >= 0.0
    assert result.tenting_area >= 0.0
    assert result.tenting_volume >= 0.0
    assert result.annulus_perimeter > 0.0
    assert result.a_point[0] > result.p_point[0]
