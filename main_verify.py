from __future__ import annotations

import json

import numpy as np

from mv_engine.core.data_models import PatientValveData
from mv_engine.pipeline.mv_analyzer import MitralValveAnalyzer


def _build_demo_data() -> PatientValveData:
    annulus = np.array(
        [
            [1.0, 0.0, 0.10],
            [0.0, 1.0, -0.10],
            [-1.0, 0.0, 0.05],
            [0.0, -1.0, -0.05],
            [0.70, 0.70, 0.08],
            [-0.70, 0.70, -0.08],
            [-0.70, -0.70, 0.03],
            [0.70, -0.70, -0.03],
        ],
        dtype=np.float64,
    )
    anterior = np.array(
        [
            [0.80, 0.15, 0.40],
            [0.60, 0.20, 0.65],
            [0.25, 0.05, 0.80],
            [0.10, 0.00, 0.90],
        ],
        dtype=np.float64,
    )
    posterior = np.array(
        [
            [-0.80, -0.20, 0.30],
            [-0.55, -0.15, 0.55],
            [-0.20, -0.05, 0.75],
            [0.00, 0.00, 0.85],
        ],
        dtype=np.float64,
    )
    return PatientValveData(
        mask_anterior=anterior,
        mask_posterior=posterior,
        mask_annulus=annulus,
    )


def main() -> None:
    analyzer = MitralValveAnalyzer(config={"voxel_size": 0.0, "fourier_degree": 2})
    result = analyzer.run_quantification(_build_demo_data())
    payload = {
        "a_point": result.a_point.tolist(),
        "p_point": result.p_point.tolist(),
        "al_commissure": result.al_commissure.tolist(),
        "pm_commissure": result.pm_commissure.tolist(),
        "tenting_point": result.tenting_point.tolist(),
        "commissure_line_dir": result.commissure_line_dir.tolist(),
        "ap_line_dir": result.ap_line_dir.tolist(),
        "plane_normal": result.annulus_plane.normal.tolist(),
        "tenting_height": result.tenting_height,
        "tenting_area": result.tenting_area,
        "tenting_volume": result.tenting_volume,
        "annulus_perimeter": result.annulus_perimeter,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
