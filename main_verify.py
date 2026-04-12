from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass

import numpy as np

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


def _build_synthetic_label_volume() -> np.ndarray:
    volume = np.zeros((128, 128, 128), dtype=np.uint8)
    center = np.array([64.0, 64.0, 64.0], dtype=np.float64)

    theta = np.linspace(-np.pi, np.pi, 360, endpoint=False, dtype=np.float64)
    mv_annulus = np.column_stack(
        (
            center[0] + 22.0 * np.cos(theta),
            center[1] + 16.0 * np.sin(theta),
            center[2] + 3.0 * np.cos(2.0 * theta),
        )
    ).astype(np.float64)

    phi = np.linspace(-np.pi, np.pi, 180, endpoint=False, dtype=np.float64)
    aortic_annulus = np.column_stack(
        (
            center[0] + 10.0 + 7.0 * np.cos(phi),
            center[1] + 4.0 * np.sin(phi),
            center[2] + 6.0 + 1.5 * np.cos(phi),
        )
    ).astype(np.float64)

    u_values = np.linspace(0.0, 1.0, 40, dtype=np.float64)
    v_values = np.linspace(-1.0, 1.0, 40, dtype=np.float64)
    anterior_points: list[np.ndarray] = []
    posterior_points: list[np.ndarray] = []

    for u_value in u_values:
        for v_value in v_values:
            anterior_points.append(
                np.array(
                    [
                        center[0] + 18.0 * (1.0 - u_value) + 0.35 * u_value,
                        center[1] + 10.0 * v_value * (1.0 - 0.15 * u_value),
                        center[2] - 7.0 * u_value - 1.4 * (1.0 - v_value * v_value),
                    ],
                    dtype=np.float64,
                )
            )
            posterior_points.append(
                np.array(
                    [
                        center[0] - 16.0 * (1.0 - u_value) - 0.35 * u_value,
                        center[1] + 9.5 * v_value * (1.0 - 0.15 * u_value),
                        center[2] - 7.0 * u_value - 1.3 * (1.0 - v_value * v_value),
                    ],
                    dtype=np.float64,
                )
            )

    _stamp_points(volume, mv_annulus, LABEL_MV_ANNULUS)
    _stamp_points(volume, aortic_annulus, LABEL_AORTIC_ANNULUS)
    _stamp_points(volume, np.vstack(anterior_points).astype(np.float64), LABEL_ANTERIOR)
    _stamp_points(volume, np.vstack(posterior_points).astype(np.float64), LABEL_POSTERIOR)
    return volume


def _to_serializable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {key: _to_serializable(field_value) for key, field_value in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _to_serializable(field_value) for key, field_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def _build_result_summary(result: object) -> dict[str, object]:
    result_dict = _to_serializable(result)
    assert isinstance(result_dict, dict)

    landmarks = result_dict["landmarks"]
    assert isinstance(landmarks, dict)
    mv_annulus = landmarks["mv_annulus"]
    av_annulus = landmarks["av_annulus"]
    assert isinstance(mv_annulus, dict)
    assert isinstance(av_annulus, dict)

    return {
        "landmarks": {
            "a_point": landmarks["a_point"],
            "p_point": landmarks["p_point"],
            "al_commissure": landmarks["al_commissure"],
            "pm_commissure": landmarks["pm_commissure"],
            "left_trigone": landmarks["left_trigone"],
            "right_trigone": landmarks["right_trigone"],
            "tenting_point": landmarks["tenting_point"],
            "commissure_line_dir": landmarks["commissure_line_dir"],
            "ap_line_dir": landmarks["ap_line_dir"],
            "mv_plane_normal": mv_annulus["plane"]["normal"],
            "av_plane_normal": av_annulus["plane"]["normal"],
            "mv_curve_samples": len(mv_annulus["curve"]),
            "av_curve_samples": len(av_annulus["curve"]),
        },
        "annulus_metrics": result_dict["annulus_metrics"],
        "shape_metrics": result_dict["shape_metrics"],
        "coaptation_metrics": result_dict["coaptation_metrics"],
        "leaflet_metrics": result_dict["leaflet_metrics"],
        "misc_metrics": result_dict["misc_metrics"],
    }


def main() -> None:
    analyzer = MitralValveAnalyzer(config={"voxel_size": 1.0, "fourier_degree": 3})

    if len(sys.argv) > 1:
        result = analyzer.run_quantification_from_nrrd(sys.argv[1])
    else:
        label_volume = build_label_volume_from_numpy(_build_synthetic_label_volume())
        patient_data = build_patient_valve_data_from_label_volume(label_volume)
        result = analyzer.run_quantification(patient_data)

    print(json.dumps(_build_result_summary(result), indent=2))


if __name__ == "__main__":
    main()
