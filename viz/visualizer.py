from __future__ import annotations

import numpy as np

from mv_engine.core.data_models import MitralValveQuantificationResult, PatientValveData


def summarize_for_debug(raw_data: PatientValveData, result: MitralValveQuantificationResult) -> dict[str, object]:
    return {
        "anterior_count": int(raw_data.mask_anterior.shape[0]),
        "posterior_count": int(raw_data.mask_posterior.shape[0]),
        "annulus_count": int(raw_data.mask_annulus.shape[0]),
        "aortic_annulus_count": int(raw_data.mask_aortic_annulus.shape[0]),
        "a_point": np.round(result.landmarks.a_point, 4).tolist(),
        "p_point": np.round(result.landmarks.p_point, 4).tolist(),
        "tenting_point": np.round(result.landmarks.tenting_point, 4).tolist(),
        "ap_diameter": round(result.annulus_metrics.ap_diameter, 4),
        "tenting_volume_ml": round(result.shape_metrics.tenting_volume_ml, 4),
    }
