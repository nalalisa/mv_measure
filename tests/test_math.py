from __future__ import annotations

import numpy as np

from mv_engine.math.fourier import build_fourier_design_matrix, fit_fourier_series, reconstruct_fourier_curve
from mv_engine.math.geometry import fit_plane_pca, point_plane_distance


def test_fit_plane_pca_returns_unit_normal() -> None:
    points = np.array(
        [
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    plane = fit_plane_pca(points)
    assert np.isclose(np.linalg.norm(plane.normal), 1.0)
    assert np.isclose(abs(plane.normal[2]), 1.0)
    assert np.isclose(point_plane_distance(np.array([0.5, 0.5, 2.0], dtype=np.float64), plane), 0.0)


def test_build_fourier_design_matrix_shape() -> None:
    theta = np.array([0.0, np.pi / 2.0], dtype=np.float64)
    design = build_fourier_design_matrix(theta, degree=2)
    assert design.shape == (2, 5)


def test_fit_fourier_series_constant_radius() -> None:
    theta = np.linspace(-np.pi, np.pi, 16, endpoint=False)
    z = np.full_like(theta, 2.0)
    coeffs = fit_fourier_series(theta, z, degree=2)
    assert np.isclose(coeffs[0], 2.0, atol=1e-10)
    assert np.allclose(coeffs[1:], 0.0, atol=1e-10)


def test_reconstruct_fourier_curve_matches_signal() -> None:
    theta = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=np.float64)
    z = 1.5 + 0.25 * np.cos(theta) - 0.5 * np.sin(2.0 * theta)
    coeffs = fit_fourier_series(theta, z, degree=2)
    reconstructed = reconstruct_fourier_curve(coeffs, theta, degree=2)
    assert np.allclose(reconstructed, z, atol=1e-10)
