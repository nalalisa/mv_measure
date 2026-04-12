from __future__ import annotations

import numpy as np

from mv_engine.core.types import FloatArray


def build_fourier_design_matrix(theta: FloatArray, degree: int) -> FloatArray:
    n_points = int(theta.shape[0])
    design_matrix = np.ones((n_points, 1 + 2 * degree), dtype=np.float64)

    for order in range(1, degree + 1):
        design_matrix[:, 2 * order - 1] = np.cos(order * theta)
        design_matrix[:, 2 * order] = np.sin(order * theta)

    return design_matrix


def fit_fourier_series(theta: FloatArray, z: FloatArray, degree: int = 3) -> FloatArray:
    design_matrix = build_fourier_design_matrix(theta, degree)
    coeffs, _, _, _ = np.linalg.lstsq(design_matrix, z, rcond=None)
    return coeffs.astype(np.float64)


def reconstruct_fourier_curve(coeffs: FloatArray, theta: FloatArray, degree: int = 3) -> FloatArray:
    z_reconstructed = np.full(theta.shape, coeffs[0], dtype=np.float64)
    for order in range(1, degree + 1):
        z_reconstructed += coeffs[2 * order - 1] * np.cos(order * theta)
        z_reconstructed += coeffs[2 * order] * np.sin(order * theta)
    return z_reconstructed.astype(np.float64)
