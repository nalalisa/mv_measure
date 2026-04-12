from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mv_engine.core.data_models import LabelVolume, PatientValveData, VolumeMetadata
from mv_engine.core.types import Matrix3D, Point3D, PointCloud3D, UInt8Array, Vector3D

LABEL_BACKGROUND = 0
LABEL_ANTERIOR = 1
LABEL_POSTERIOR = 2
LABEL_MV_ANNULUS = 3
LABEL_AORTIC_ANNULUS = 4


def _require_nrrd() -> Any:
    try:
        import nrrd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "pynrrd is required for NRRD input support. Install it with `pip install pynrrd`."
        ) from exc
    return nrrd


def _parse_spacing_from_header(header: dict[str, Any]) -> Vector3D:
    spacings = header.get("spacings")
    if spacings is not None:
        return np.asarray(spacings, dtype=np.float64)

    space_directions = header.get("space directions")
    if space_directions is None:
        return np.ones(3, dtype=np.float64)

    direction_matrix = _parse_direction_matrix(header)
    return np.linalg.norm(direction_matrix, axis=1).astype(np.float64)


def _parse_direction_matrix(header: dict[str, Any]) -> Matrix3D:
    raw = header.get("space directions")
    if raw is None:
        spacing = header.get("spacings")
        if spacing is None:
            return np.eye(3, dtype=np.float64)
        return np.diag(np.asarray(spacing, dtype=np.float64))

    rows: list[np.ndarray] = []
    for item in raw:
        if item is None:
            rows.append(np.zeros(3, dtype=np.float64))
            continue
        rows.append(np.asarray(item, dtype=np.float64))
    return np.vstack(rows).astype(np.float64)


def _parse_origin(header: dict[str, Any]) -> Point3D:
    raw_origin = header.get("space origin")
    if raw_origin is None:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(raw_origin, dtype=np.float64)


def load_labeled_volume_from_nrrd(file_path: str | Path) -> LabelVolume:
    nrrd = _require_nrrd()
    labels, header = nrrd.read(str(file_path))
    if labels.ndim != 3:
        raise ValueError("Only 3D NRRD label volumes are supported.")

    labels_u8 = np.asarray(labels, dtype=np.uint8)
    direction_matrix = _parse_direction_matrix(header)
    metadata = VolumeMetadata(
        spacing=_parse_spacing_from_header(header),
        origin=_parse_origin(header),
        direction_matrix=direction_matrix,
        shape=(
            int(labels_u8.shape[0]),
            int(labels_u8.shape[1]),
            int(labels_u8.shape[2]),
        ),
    )
    return LabelVolume(labels=labels_u8, metadata=metadata)


def extract_label_points(label_volume: LabelVolume, label_value: int) -> PointCloud3D:
    voxel_indices = np.argwhere(label_volume.labels == label_value).astype(np.float64)
    if voxel_indices.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    world_points = voxel_indices @ label_volume.metadata.direction_matrix
    world_points += label_volume.metadata.origin
    return world_points.astype(np.float64)


def build_patient_valve_data_from_label_volume(label_volume: LabelVolume) -> PatientValveData:
    return PatientValveData(
        mask_anterior=extract_label_points(label_volume, LABEL_ANTERIOR),
        mask_posterior=extract_label_points(label_volume, LABEL_POSTERIOR),
        mask_annulus=extract_label_points(label_volume, LABEL_MV_ANNULUS),
        mask_aortic_annulus=extract_label_points(label_volume, LABEL_AORTIC_ANNULUS),
        metadata=label_volume.metadata,
    )


def load_patient_valve_data_from_nrrd(file_path: str | Path) -> PatientValveData:
    label_volume = load_labeled_volume_from_nrrd(file_path)
    return build_patient_valve_data_from_label_volume(label_volume)


def build_label_volume_from_numpy(
    labels: UInt8Array,
    spacing: Vector3D | None = None,
    origin: Point3D | None = None,
    direction_matrix: Matrix3D | None = None,
) -> LabelVolume:
    spacing_value = np.ones(3, dtype=np.float64) if spacing is None else np.asarray(spacing, dtype=np.float64)
    origin_value = np.zeros(3, dtype=np.float64) if origin is None else np.asarray(origin, dtype=np.float64)
    direction_value = (
        np.diag(spacing_value).astype(np.float64)
        if direction_matrix is None
        else np.asarray(direction_matrix, dtype=np.float64)
    )
    metadata = VolumeMetadata(
        spacing=spacing_value,
        origin=origin_value,
        direction_matrix=direction_value,
        shape=(int(labels.shape[0]), int(labels.shape[1]), int(labels.shape[2])),
    )
    return LabelVolume(labels=np.asarray(labels, dtype=np.uint8), metadata=metadata)
