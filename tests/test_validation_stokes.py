# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

import numpy as np
from fluidflow.models.oscillatory_bl import OscillatoryBLModel


# ---------------------------------------------------------------------------
# Stokes-layer utilities
# ---------------------------------------------------------------------------

def stokes_domain_height_ok(H, nu, omega, eps_bl=1e-6, beta=25.0):
    """Returns True if domain height H is sufficient for Stokes validation."""
    delta = np.sqrt(2 * nu / omega)
    return H >= max(delta * np.log(1 / eps_bl), beta * delta)


def stokes_far_field_diagnostic(z, u, t, nu, omega, alpha=6.0):
    """Detects far-field contamination in oscillatory Stokes flow.

    Parameters
    ----------
    z : array (Nz,)
    u : array (Nz,)
    t : float
    nu : float
    omega : float
    alpha : float
        Far-field cutoff in units of delta.

    Returns
    -------
    dict with amp_error, shear_error, z_ff, delta.
    """
    delta = np.sqrt(2 * nu / omega)
    z_ff = alpha * delta

    mask = z >= z_ff
    if not np.any(mask):
        raise ValueError("Domain too short to define far-field region")

    u_ff = u[mask]
    z_ff_vals = z[mask]

    u_inf = -np.cos(t)
    amp_error = np.max(np.abs(u_ff - u_inf))

    du_dz = np.gradient(u_ff, z_ff_vals)
    shear_error = np.max(np.abs(du_dz))

    return {
        "amp_error": amp_error,
        "shear_error": shear_error,
        "z_ff": z_ff,
        "delta": delta,
    }


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def test_stokes_laminar_profile():
    """Laminar solver (turbulence_enabled=False) should match analytic Stokes solution.

    The model solves u_t = F0*sin(omega*t) + nu*u_zz with no-slip at z=0
    and free-slip at z=H. The analytic periodic steady-state solution for
    a body-force-driven oscillatory boundary layer is:

        u(z,t) = -cos(t) + exp(-z/delta)*cos(t - z/delta)

    where delta = sqrt(2*nu/omega). This test validates the laminar solver
    (molecular diffusion only, no turbulence closure).
    """
    Re = 100
    nu = 1.0 / Re
    omega = 1.0
    delta = np.sqrt(2 * nu / omega)

    # Domain height: max(delta*ln(1/eps), beta*delta) with beta=25
    H = 25 * delta  # ~3.54 for Re=100
    assert stokes_domain_height_ok(H, nu, omega)

    params = dict(
        Re=Re, S=0.0, Lambda=0.0,
        N=128, H=H, gamma=2.0,
        Sc_t=1.0, turbulence_enabled=False,
    )
    model = OscillatoryBLModel(params)
    z = model.grid.z

    # Initialize with the exact periodic steady-state at t=0 to avoid a
    # slow-decaying DC transient (timescale H^2/nu >> oscillation period).
    # u_analytic(z, 0) = -cos(0) + exp(-z/delta)*cos(-z/delta)
    #                   = -1 + exp(-z/delta)*cos(z/delta)
    u = -np.cos(0.0) + np.exp(-z / delta) * np.cos(0.0 - z / delta)
    u[0] = 0.0  # no-slip
    C = np.zeros(model.grid.N)
    C[0] = model.C_ref

    # Run for a few cycles — transient should be minimal with correct IC
    n_cycles = 3
    t = 0.0
    while t < n_cycles * 2 * np.pi:
        u, C = model.step(u, C, t)
        t += model.dt

    # 1. Far-field diagnostic: top of domain should be clean
    diag = stokes_far_field_diagnostic(z, u, t, nu, omega, alpha=6.0)
    assert diag["amp_error"] < 0.01, (
        f"Far-field amplitude error {diag['amp_error']:.6f} indicates domain contamination"
    )
    assert diag["shear_error"] < 0.02, (
        f"Far-field shear error {diag['shear_error']:.6f} indicates domain contamination"
    )

    # 2. BL profile comparison against analytic solution
    u_analytic = -np.cos(t) + np.exp(-z / delta) * np.cos(t - z / delta)
    mask = z < 5 * delta
    error = np.max(np.abs(u[mask] - u_analytic[mask])) / np.max(np.abs(u_analytic[mask]))
    assert error < 0.05, f"Stokes profile error {error:.4f} exceeds 5%"
