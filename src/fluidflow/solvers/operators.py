# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT


import numpy as np
from numba import njit

def grad_central(u, dx):
    return (np.roll(u, -1) - np.roll(u, 1)) / (2*dx)

def laplacian(u, dx):
    return (np.roll(u, -1) - 2*u + np.roll(u, 1)) / dx**2


# ---------------------------------------------------------------------------
# Stretched-grid finite difference operators
# ---------------------------------------------------------------------------

@njit(cache=True)
def ddz_impl(f, z, N):
    """Numba-compiled first derivative on a stretched grid."""
    df = np.empty(N)

    # Interior: 2nd-order central difference on non-uniform grid
    for i in range(1, N - 1):
        h_p = z[i + 1] - z[i]
        h_m = z[i] - z[i - 1]
        df[i] = (
            h_m * h_m * f[i + 1] + (h_p * h_p - h_m * h_m) * f[i] - h_p * h_p * f[i - 1]
        ) / (h_m * h_p * (h_m + h_p))

    # Left boundary (i=0): forward 3-point
    h0 = z[1] - z[0]
    h1 = z[2] - z[1]
    df[0] = (
        -(2 * h0 + h1) / (h0 * (h0 + h1)) * f[0]
        + (h0 + h1) / (h0 * h1) * f[1]
        - h0 / (h1 * (h0 + h1)) * f[2]
    )

    # Right boundary (i=N-1): backward 3-point
    h_n1 = z[N - 1] - z[N - 2]
    h_n2 = z[N - 2] - z[N - 3]
    df[N - 1] = (
        h_n1 / (h_n2 * (h_n2 + h_n1)) * f[N - 3]
        - (h_n2 + h_n1) / (h_n2 * h_n1) * f[N - 2]
        + (2 * h_n1 + h_n2) / (h_n1 * (h_n2 + h_n1)) * f[N - 1]
    )

    return df


def ddz(f, grid):
    """First derivative df/dz on a stretched grid.

    Uses second-order non-uniform central differences in physical space
    for interior points, and second-order one-sided stencils at boundaries.

    Exact for linear functions, second-order accurate in general.
    """
    return ddz_impl(f, grid.z, grid.N)


def d2dz2(f, grid):
    """Second derivative d2f/dz2 on a stretched grid.

    Uses second-order non-uniform central differences in physical space
    for interior points, and second-order one-sided stencils at boundaries.

    Exact for quadratic functions, second-order accurate in general.
    """
    N = grid.N
    z = grid.z
    d2f = np.empty(N)

    # Interior: standard 3-point non-uniform second derivative
    h_p = z[2:] - z[1:-1]
    h_m = z[1:-1] - z[:-2]
    d2f[1:-1] = 2.0 * (
        h_m * f[2:] - (h_m + h_p) * f[1:-1] + h_p * f[:-2]
    ) / (h_m * h_p * (h_m + h_p))

    # Boundary: 2nd-order one-sided (3-point) stencils
    # Left boundary (i=0): forward stencil using z[0], z[1], z[2]
    h0 = z[1] - z[0]
    h1 = z[2] - z[1]
    d2f[0] = 2.0 * (h1*f[0] - (h0+h1)*f[1] + h0*f[2]) / (h0 * h1 * (h0 + h1))

    # Right boundary (i=N-1): backward stencil using z[-3], z[-2], z[-1]
    hm2 = z[-1] - z[-2]
    hm1 = z[-2] - z[-3]
    d2f[-1] = 2.0 * (hm1*f[-1] - (hm2+hm1)*f[-2] + hm2*f[-3]) / (hm2 * hm1 * (hm2 + hm1))

    return d2f


def diffusion_operator_matrix(nu, grid):
    """Build matrix L such that L @ f = d/dz[nu(z) d/dz f].

    Returns a dense (N, N) matrix for variable nu(z).
    Uses a conservative finite-volume discretization on the non-uniform grid:
        (L f)_i = (1/dz_c_i) * [nu_{i+1/2}/h_p * (f_{i+1}-f_i)
                                - nu_{i-1/2}/h_m * (f_i-f_{i-1})]
    where h_p, h_m are forward/backward spacings and dz_c is the control
    volume width (h_m + h_p)/2.

    Boundary rows are zero (caller sets BCs separately).
    """
    N = grid.N
    z = grid.z

    L = np.zeros((N, N))
    for i in range(1, N - 1):
        h_m = z[i] - z[i - 1]
        h_p = z[i + 1] - z[i]
        dz_c = 0.5 * (h_m + h_p)

        nu_half_m = 0.5 * (nu[i] + nu[i - 1])
        nu_half_p = 0.5 * (nu[i] + nu[i + 1])

        L[i, i - 1] = nu_half_m / (h_m * dz_c)
        L[i, i]     = -(nu_half_m / h_m + nu_half_p / h_p) / dz_c
        L[i, i + 1] = nu_half_p / (h_p * dz_c)

    return L


@njit(cache=True)
def diffusion_operator_bands(nu, z, N):
    """Build tridiagonal bands for L such that L @ f = d/dz[nu(z) d/dz f].

    Returns (lower, diag, upper) arrays of length N, with boundary rows = 0.
    This avoids building the full N x N matrix.
    """
    lower = np.zeros(N)
    diag = np.zeros(N)
    upper = np.zeros(N)

    for i in range(1, N - 1):
        h_m = z[i] - z[i - 1]
        h_p = z[i + 1] - z[i]
        dz_c = 0.5 * (h_m + h_p)

        nu_half_m = 0.5 * (nu[i] + nu[i - 1])
        nu_half_p = 0.5 * (nu[i] + nu[i + 1])

        lower[i] = nu_half_m / (h_m * dz_c)
        diag[i] = -(nu_half_m / h_m + nu_half_p / h_p) / dz_c
        upper[i] = nu_half_p / (h_p * dz_c)

    return lower, diag, upper
