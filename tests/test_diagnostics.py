# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

# tests/test_diagnostics.py
import numpy as np
from fluidflow.grid import StretchedGrid
from fluidflow.diagnostics import CycleDiagnostics


def test_diagnostics_from_constant_fields():
    """With constant u and C, diagnostics should be well-defined."""
    g = StretchedGrid(N=64, H=1.0, gamma=2.0)
    diag = CycleDiagnostics(g)

    u = np.ones(g.N)
    C = np.zeros(g.N)
    nu_t = np.ones(g.N) * 0.01
    D_t = nu_t / 1.0
    nu = 0.001

    diag.accumulate(u, C, nu_t, D_t, nu, t=0.0)
    diag.accumulate(u, C, nu_t, D_t, nu, t=0.5)

    result = diag.finalize()
    assert "viscosity_ratio" in result
    assert "drag_coefficient" in result
    assert "kinetic_energy" in result
    assert "sediment_flux" in result


def test_classification_turbulent():
    """High nu_t / nu should classify as turbulent."""
    g = StretchedGrid(N=64, H=1.0, gamma=2.0)
    diag = CycleDiagnostics(g)

    nu = 0.001
    nu_t = np.ones(g.N) * 0.1  # ratio = 100
    diag.accumulate(np.ones(g.N), np.zeros(g.N), nu_t, nu_t, nu, t=0.0)
    result = diag.finalize()
    assert result["regime"] == "turbulent"


def test_classification_laminar():
    """Low nu_t / nu should classify as laminar."""
    g = StretchedGrid(N=64, H=1.0, gamma=2.0)
    diag = CycleDiagnostics(g)

    nu = 0.01
    nu_t = np.ones(g.N) * 1e-5  # ratio = 0.001
    diag.accumulate(np.ones(g.N), np.zeros(g.N), nu_t, nu_t, nu, t=0.0)
    result = diag.finalize()
    assert result["regime"] == "laminar"


def test_finalize_includes_profiles():
    """Finalized diagnostics include cycle-averaged vertical profiles."""
    g = StretchedGrid(N=64, H=1.0, gamma=2.0)
    diag = CycleDiagnostics(g, g_prime=0.5)

    u = np.linspace(0.0, 1.0, g.N)
    C = np.linspace(1.0, 0.0, g.N)
    nu_t = np.ones(g.N) * 0.01
    D_t = nu_t / 1.0
    nu = 0.001

    diag.accumulate(u, C, nu_t, D_t, nu, t=0.0)
    diag.accumulate(u, C, nu_t, D_t, nu, t=0.5)

    result = diag.finalize()
    assert "profiles" in result
    for key in ("nu_t", "C", "Ri_g", "z"):
        assert key in result["profiles"], f"Missing profile: {key}"
    assert len(result["profiles"]["z"]) == g.N


def test_finalize_profiles_default_g_prime():
    """Without g_prime, profiles should still be present with Ri_g=0."""
    g = StretchedGrid(N=32, H=1.0, gamma=0.0)
    diag = CycleDiagnostics(g)

    u = np.ones(g.N)
    C = np.zeros(g.N)
    nu_t = np.ones(g.N) * 0.01
    D_t = nu_t
    nu = 0.001

    diag.accumulate(u, C, nu_t, D_t, nu, t=0.0)
    result = diag.finalize()
    assert "profiles" in result
    assert np.allclose(result["profiles"]["Ri_g"], 0.0)


def test_viscosity_ratio_uses_trapezoidal_integration():
    """Viscosity ratio must use trapezoidal integration over z, not np.mean.

    On a stretched grid (gamma > 0) with nu_t = z^2, grid points cluster
    near z=0 where z^2 is small. A naive np.mean over grid points
    overweights the near-bed region and underestimates the domain average.
    The correct domain average is the trapezoidal integral divided by domain
    height.
    """
    g = StretchedGrid(N=64, H=5.0, gamma=2.0)
    nu = 0.001

    # Synthetic nu_t profile: z^2, which grows away from the bed.
    nu_t_profile = g.z ** 2

    # Sanity check: on a stretched grid these two averages differ
    naive_mean = np.mean(nu_t_profile)
    trapz_mean = np.trapezoid(nu_t_profile, g.z) / g.z[-1]
    assert not np.isclose(naive_mean, trapz_mean, rtol=0.01), (
        "Test setup error: naive mean and trapezoidal mean should differ "
        "on a stretched grid with nu_t = z^2"
    )

    # Build CycleDiagnostics and feed the synthetic profile
    diag = CycleDiagnostics(g)
    u = np.ones(g.N)       # constant velocity (not used for viscosity_ratio)
    C = np.zeros(g.N)      # zero concentration
    D_t = nu_t_profile / 1.0

    diag.accumulate(u, C, nu_t_profile, D_t, nu, t=0.0)
    result = diag.finalize()

    expected = np.trapezoid(nu_t_profile, g.z) / g.z[-1] / nu
    assert np.isclose(result["viscosity_ratio"], expected, rtol=1e-12), (
        f"viscosity_ratio {result['viscosity_ratio']} does not match "
        f"trapezoidal integral {expected}"
    )
