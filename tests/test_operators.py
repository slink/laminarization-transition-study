# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

import numpy as np
from fluidflow.grid import StretchedGrid
from fluidflow.solvers.operators import (
    grad_central, laplacian,  # existing periodic operators
    ddz, d2dz2, diffusion_operator_matrix, diffusion_operator_bands,
)

def test_ddz_linear():
    """d/dz of a linear function z should be 1 everywhere (interior)."""
    g = StretchedGrid(N=64, H=1.0, gamma=2.0)
    f = g.z.copy()
    df = ddz(f, g)
    assert np.allclose(df[1:-1], 1.0, atol=1e-10)

def test_ddz_quadratic():
    """d/dz of z^2 should be 2z."""
    g = StretchedGrid(N=128, H=1.0, gamma=2.0)
    f = g.z**2
    df = ddz(f, g)
    expected = 2.0 * g.z
    assert np.allclose(df[1:-1], expected[1:-1], rtol=1e-4)

def test_d2dz2_quadratic():
    """d^2/dz^2 of z^2 should be 2."""
    g = StretchedGrid(N=128, H=1.0, gamma=2.0)
    f = g.z**2
    d2f = d2dz2(f, g)
    assert np.allclose(d2f[1:-1], 2.0, rtol=1e-3)

def test_ddz_convergence():
    """Derivative of sin(pi*z/H) should converge at 2nd order."""
    H = 1.0
    errors = []
    for N in [64, 128, 256]:
        g = StretchedGrid(N=N, H=H, gamma=2.0)
        f = np.sin(np.pi * g.z / H)
        df = ddz(f, g)
        exact = (np.pi / H) * np.cos(np.pi * g.z / H)
        errors.append(np.max(np.abs(df[2:-2] - exact[2:-2])))
    ratio = errors[0] / errors[1]
    assert ratio > 3.0

def test_diffusion_matrix_shape():
    """Diffusion matrix should be (N, N) tridiagonal."""
    g = StretchedGrid(N=32, H=1.0, gamma=2.0)
    nu = np.ones(g.N) * 0.1
    A = diffusion_operator_matrix(nu, g)
    assert A.shape == (32, 32)

def test_diffusion_matrix_constant_nu():
    """With constant nu, L*z^2 should give 2*nu everywhere interior."""
    g = StretchedGrid(N=128, H=1.0, gamma=2.0)
    nu_val = 0.5
    nu = np.full(g.N, nu_val)
    A = diffusion_operator_matrix(nu, g)
    f = g.z**2
    Lf = A @ f
    assert np.allclose(Lf[2:-2], 2.0 * nu_val, rtol=1e-2)


def test_diffusion_bands_cross_validate():
    """Bands from diffusion_operator_bands must match diagonals of the full matrix.

    Tests constant, linear, and random nu profiles across several grid sizes.
    """
    rng = np.random.default_rng(42)

    for N in [32, 64, 128]:
        g = StretchedGrid(N=N, H=1.0, gamma=2.0)

        nu_profiles = {
            "constant": np.full(N, 0.3),
            "linear": np.linspace(0.1, 1.0, N),
            "random": rng.uniform(0.01, 1.0, size=N),
        }

        for label, nu in nu_profiles.items():
            A = diffusion_operator_matrix(nu, g)
            lower, diag, upper = diffusion_operator_bands(nu, g.z, g.N)

            # Extract diagonals from the full matrix
            mat_diag = np.diag(A, 0)
            mat_lower = np.diag(A, -1)  # length N-1
            mat_upper = np.diag(A, 1)   # length N-1

            msg = f"N={N}, nu={label}"
            assert np.allclose(diag, mat_diag, atol=1e-14), (
                f"Main diagonal mismatch ({msg})"
            )
            # lower[i] corresponds to A[i, i-1]; for i=1..N-1,
            # np.diag(A, -1)[i-1] = A[i, i-1], so mat_lower[i-1] == lower[i].
            assert np.allclose(lower[1:], mat_lower, atol=1e-14), (
                f"Lower diagonal mismatch ({msg})"
            )
            # upper[i] corresponds to A[i, i+1]; for i=0..N-2,
            # np.diag(A, 1)[i] = A[i, i+1], so mat_upper[i] == upper[i].
            assert np.allclose(upper[:-1], mat_upper, atol=1e-14), (
                f"Upper diagonal mismatch ({msg})"
            )


def test_diffusion_bands_constant_nu():
    """With constant nu, applying bands as tridiagonal op to z^2 gives 2*nu."""
    g = StretchedGrid(N=128, H=1.0, gamma=2.0)
    nu_val = 0.5
    nu = np.full(g.N, nu_val)
    lower, diag, upper = diffusion_operator_bands(nu, g.z, g.N)

    f = g.z ** 2

    # Apply tridiagonal operator: (L f)_i = lower[i]*f[i-1] + diag[i]*f[i] + upper[i]*f[i+1]
    Lf = np.zeros(g.N)
    for i in range(1, g.N - 1):
        Lf[i] = lower[i] * f[i - 1] + diag[i] * f[i] + upper[i] * f[i + 1]

    # Interior points (skip boundaries) should equal 2*nu
    assert np.allclose(Lf[2:-2], 2.0 * nu_val, rtol=1e-2)
