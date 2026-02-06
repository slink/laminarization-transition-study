# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT


import numpy as np
from numba import njit
from fluidflow.solvers.operators import ddz, ddz_impl

KAPPA = 0.41  # von Karman constant


def ri_damping(Ri_g, Ri_c=0.25, damping="linear"):
    """Stratification damping function.

    Parameters
    ----------
    Ri_g : float
        Gradient Richardson number.
    Ri_c : float
        Critical Richardson number (default 0.25).
    damping : str
        "linear"      -> f = max(0, 1 - Ri_g/Ri_c)
        "exponential" -> f = exp(-Ri_g / Ri_c)
    """
    if damping == "linear":
        return max(0.0, 1.0 - Ri_g / Ri_c)
    elif damping == "exponential":
        import math
        return math.exp(-Ri_g / Ri_c)
    else:
        raise ValueError(f"Unknown damping type: {damping}")


def mixing_length_nu_t(dudz, z, kappa=KAPPA):
    """Mixing-length turbulent viscosity (no stratification correction).

    nu_t0 = kappa^2 * z^2 * |du/dz|
    """
    return kappa**2 * z**2 * np.abs(dudz)


@njit(cache=True)
def _compute_nu_t_linear(u, C, z, N, g_prime, Sc_t, Ri_c, epsilon, kappa):
    """JIT-compiled core of compute_nu_t for linear damping."""
    dudz_arr = ddz_impl(u, z, N)
    dCdz = ddz_impl(C, z, N)

    nu_t = np.empty(N)
    D_t = np.empty(N)

    kappa_sq = kappa * kappa
    for i in range(N):
        # Mixing length
        nu_t0 = kappa_sq * z[i] * z[i] * abs(dudz_arr[i])

        # Richardson number with safe divide
        dudz_sq = dudz_arr[i] * dudz_arr[i]
        if dudz_sq > epsilon:
            Ri_g = -g_prime * dCdz[i] / dudz_sq
        else:
            Ri_g = 0.0

        # Linear damping: f = clip(1 - Ri_g/Ri_c, 0, 1)
        f_Ri = 1.0 - Ri_g / Ri_c
        if f_Ri < 0.0:
            f_Ri = 0.0
        elif f_Ri > 1.0:
            f_Ri = 1.0

        nu_t[i] = nu_t0 * f_Ri
        D_t[i] = nu_t[i] / Sc_t

    return nu_t, D_t


@njit(cache=True)
def _compute_nu_t_exponential(u, C, z, N, g_prime, Sc_t, Ri_c, epsilon, kappa):
    """JIT-compiled core of compute_nu_t for exponential damping."""
    dudz_arr = ddz_impl(u, z, N)
    dCdz = ddz_impl(C, z, N)

    nu_t = np.empty(N)
    D_t = np.empty(N)

    kappa_sq = kappa * kappa
    Ri_max = 50.0 * Ri_c
    for i in range(N):
        nu_t0 = kappa_sq * z[i] * z[i] * abs(dudz_arr[i])

        dudz_sq = dudz_arr[i] * dudz_arr[i]
        if dudz_sq > epsilon:
            Ri_g = -g_prime * dCdz[i] / dudz_sq
        else:
            Ri_g = 0.0

        # Exponential damping: f = exp(-clip(Ri_g, 0, 50*Ri_c) / Ri_c)
        Ri_clamped = Ri_g
        if Ri_clamped < 0.0:
            Ri_clamped = 0.0
        elif Ri_clamped > Ri_max:
            Ri_clamped = Ri_max
        f_Ri = np.exp(-Ri_clamped / Ri_c)

        nu_t[i] = nu_t0 * f_Ri
        D_t[i] = nu_t[i] / Sc_t

    return nu_t, D_t


def compute_nu_t(u, C, grid, g_prime, Sc_t=1.0, Ri_c=0.25, epsilon=1e-10,
                 damping="linear"):
    """Full turbulent viscosity with stratification damping.

    Steps:
        1. Compute du/dz and dC/dz via the stretched-grid ddz operator.
        2. Compute base mixing-length viscosity: nu_t0 = kappa^2 z^2 |du/dz|.
        3. Compute gradient Richardson number:
              Ri_g = -g_prime * dC/dz / (du/dz^2 + epsilon)
        4. Apply damping: nu_t = nu_t0 * f(Ri_g).
        5. Compute turbulent diffusivity: D_t = nu_t / Sc_t.

    Parameters
    ----------
    u : ndarray, shape (N,)
        Velocity profile.
    C : ndarray, shape (N,)
        Concentration profile.
    grid : StretchedGrid
        Grid object with attribute z and compatible with ddz.
    g_prime : float
        Reduced gravity (g * Delta_rho / rho_0 or similar).
    Sc_t : float
        Turbulent Schmidt number (default 1.0).
    Ri_c : float
        Critical Richardson number (default 0.25).
    epsilon : float
        Small regularisation to avoid division by zero in Ri_g.
    damping : str
        "linear" or "exponential" (see ri_damping()).

    Returns
    -------
    nu_t : ndarray, shape (N,)
        Turbulent viscosity.
    D_t : ndarray, shape (N,)
        Turbulent diffusivity.
    """
    if damping == "linear":
        return _compute_nu_t_linear(u, C, grid.z, grid.N, g_prime, Sc_t,
                                    Ri_c, epsilon, KAPPA)
    elif damping == "exponential":
        return _compute_nu_t_exponential(u, C, grid.z, grid.N, g_prime, Sc_t,
                                         Ri_c, epsilon, KAPPA)
    else:
        raise ValueError(f"Unknown damping type: {damping}")
