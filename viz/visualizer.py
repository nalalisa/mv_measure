from __future__ import annotations

import numpy as np

from mv_engine.core.data_models import MitralLandmarks, PatientValveData


def summarize_for_debug(raw_data: PatientValveData, landmarks: MitralLandmarks) -> dict[str, object]:
    return {
        "anterior_count": int(raw_data.mask_anterior.shape[0]),
        "posterior_count": int(raw_data.mask_posterior.shape[0]),
        "annulus_count": int(raw_data.mask_annulus.shape[0]),
        "a_point": np.round(landmarks.a_point, 4).tolist(),
        "p_point": np.round(landmarks.p_point, 4).tolist(),
        "tenting_point": np.round(landmarks.tenting_point, 4).tolist(),
    }
