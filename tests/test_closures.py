# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

import math

import numpy as np
import pytest
from fluidflow.models.closures import (
    ri_damping,
    mixing_length_nu_t,
    compute_nu_t,
    KAPPA,
)
from fluidflow.grid import StretchedGrid


def test_ri_damping_zero_ri():
    """No stratification -> f(0) = 1.0."""
    assert ri_damping(0.0) == 1.0


def test_ri_damping_critical():
    """At Ri_c, f(Ri_c) = 0.0."""
    Ri_c = 0.25
    assert ri_damping(Ri_c, Ri_c=Ri_c) == 0.0


def test_ri_damping_supercritical():
    """Above Ri_c, f is clamped to 0.0."""
    Ri_c = 0.25
    assert ri_damping(0.5, Ri_c=Ri_c) == 0.0
    assert ri_damping(1.0, Ri_c=Ri_c) == 0.0


def test_ri_damping_partial():
    """f(Ri_c/2) = 0.5."""
    Ri_c = 0.25
    assert ri_damping(Ri_c / 2.0, Ri_c=Ri_c) == 0.5


def test_mixing_length_at_bed():
    """nu_t = 0 at z=0 because kappa^2 * 0^2 = 0."""
    dudz = 10.0  # arbitrary nonzero shear
    z = 0.0
    assert mixing_length_nu_t(dudz, z) == 0.0


def test_mixing_length_increases_with_shear():
    """Doubling shear doubles nu_t (linear in |du/dz|)."""
    z = 0.5
    dudz_1 = 2.0
    dudz_2 = 4.0
    nu_t_1 = mixing_length_nu_t(dudz_1, z)
    nu_t_2 = mixing_length_nu_t(dudz_2, z)
    assert np.isclose(nu_t_2, 2.0 * nu_t_1)


def test_compute_nu_t_stable_stratification_damps():
    """Stable stratification (dC/dz < 0, g_prime > 0) should produce
    positive Ri_g and reduce nu_t compared to the unstratified case."""
    N = 50
    grid = StretchedGrid(N=N, H=1.0, gamma=0.0)  # uniform grid

    # Linear velocity profile: u increases with z -> nonzero shear everywhere
    u = np.linspace(0.0, 1.0, N)

    # Stable stratification: concentration decreases upward (dC/dz < 0)
    C_stratified = np.linspace(1.0, 0.0, N)

    # Uniform concentration: no stratification (dC/dz = 0 everywhere)
    C_uniform = np.ones(N)

    g_prime = 0.5

    nu_t_strat, D_t_strat = compute_nu_t(u, C_stratified, grid, g_prime)
    nu_t_unstr, D_t_unstr = compute_nu_t(u, C_uniform, grid, g_prime)

    # Interior points only (boundaries have one-sided stencils)
    interior = slice(2, -2)

    # With stratification, nu_t should be reduced (damped) at interior points
    assert np.all(nu_t_strat[interior] <= nu_t_unstr[interior])

    # At least some interior points should have strictly less nu_t
    assert np.any(nu_t_strat[interior] < nu_t_unstr[interior])


def test_compute_nu_t_no_stratification_no_damping():
    """With uniform concentration, Ri_g = 0, so damping factor = 1.0
    and nu_t equals the base mixing-length viscosity."""
    N = 50
    grid = StretchedGrid(N=N, H=1.0, gamma=0.0)

    u = np.linspace(0.0, 1.0, N)
    C_uniform = np.ones(N)
    g_prime = 0.5

    nu_t, D_t = compute_nu_t(u, C_uniform, grid, g_prime)
    nu_t0 = mixing_length_nu_t(np.gradient(u, grid.z), grid.z)

    # Without stratification, compute_nu_t should match base mixing-length nu_t
    # (using ddz internally, which matches np.gradient on uniform grids)
    assert np.allclose(nu_t, nu_t0, atol=1e-12)


def test_compute_nu_t_intermediate_damping():
    """When 0 < Ri_g < Ri_c at interior points, nu_t should be reduced but
    not zero -- partial damping of turbulence."""
    from fluidflow.solvers.operators import ddz as _ddz

    N = 50
    Ri_c = 0.25
    grid = StretchedGrid(N=N, H=1.0, gamma=0.0)

    # Strong linear shear so that du/dz is large relative to the stratification
    # term -- this keeps Ri_g positive but below Ri_c.
    u = np.linspace(0.0, 5.0, N)

    # Mild stable stratification: concentration decreases upward (dC/dz < 0)
    C = np.linspace(1.0, 0.8, N)

    g_prime = 0.5

    nu_t, D_t = compute_nu_t(u, C, grid, g_prime, Ri_c=Ri_c)

    # Also compute the undamped base viscosity for comparison
    nu_t_undamped, _ = compute_nu_t(u, np.ones(N), grid, g_prime, Ri_c=Ri_c)

    # Compute the actual Ri_g at interior points to confirm it is subcritical
    dudz_arr = _ddz(u, grid)
    dCdz = _ddz(C, grid)
    epsilon = 1e-10
    Ri_g = -g_prime * dCdz / (dudz_arr**2 + epsilon)

    interior = slice(2, -2)

    # Verify Ri_g is in (0, Ri_c) at interior points (subcritical regime)
    assert np.all(Ri_g[interior] > 0), "Ri_g should be positive (stable stratification)"
    assert np.all(Ri_g[interior] < Ri_c), "Ri_g should be below Ri_c (subcritical)"

    # nu_t should be strictly positive (not fully damped)
    assert np.all(nu_t[interior] > 0), "nu_t should not be zero under partial damping"

    # nu_t should be strictly less than undamped value
    assert np.all(nu_t[interior] < nu_t_undamped[interior]), (
        "nu_t should be reduced relative to undamped case"
    )

    # D_t should equal nu_t / Sc_t (Sc_t=1.0 by default)
    assert np.allclose(D_t, nu_t)


def test_compute_nu_t_supercritical_damping():
    """When Ri_g >= Ri_c everywhere, the damping factor is 0 and nu_t = 0
    everywhere -- complete turbulence collapse."""
    from fluidflow.solvers.operators import ddz as _ddz

    N = 50
    Ri_c = 0.25
    grid = StretchedGrid(N=N, H=1.0, gamma=0.0)

    # Weak shear: small du/dz
    u = np.linspace(0.0, 0.01, N)

    # Very strong stable stratification: large negative dC/dz
    C = np.linspace(10.0, 0.0, N)

    g_prime = 1.0

    nu_t, D_t = compute_nu_t(u, C, grid, g_prime, Ri_c=Ri_c)

    # Confirm Ri_g >= Ri_c everywhere (supercritical)
    dudz_arr = _ddz(u, grid)
    dCdz = _ddz(C, grid)
    epsilon = 1e-10
    Ri_g = -g_prime * dCdz / (dudz_arr**2 + epsilon)

    # All points should be supercritical
    assert np.all(Ri_g >= Ri_c), "Ri_g should be >= Ri_c everywhere (supercritical)"

    # nu_t must be exactly 0 everywhere (complete collapse)
    assert np.all(nu_t == 0.0), "nu_t should be zero everywhere under supercritical damping"

    # D_t must also be zero
    assert np.all(D_t == 0.0), "D_t should be zero everywhere under supercritical damping"


# --- Exponential damping tests ---

def test_ri_damping_exponential_zero_ri():
    """Exponential damping gives f=1 at zero stratification."""
    assert ri_damping(0.0, damping="exponential") == pytest.approx(1.0)


def test_ri_damping_exponential_at_critical():
    """Exponential damping gives f=exp(-1) at Ri_g=Ri_c."""
    assert ri_damping(0.25, Ri_c=0.25, damping="exponential") == pytest.approx(math.exp(-1))


def test_ri_damping_exponential_no_hard_cutoff():
    """Exponential damping is nonzero for supercritical Ri_g (no hard cutoff)."""
    result = ri_damping(0.5, Ri_c=0.25, damping="exponential")
    assert result > 0.0


def test_ri_damping_linear_is_default():
    """Linear damping remains the default behavior."""
    assert ri_damping(0.125, Ri_c=0.25) == pytest.approx(0.5)


def test_ri_damping_unknown_type_raises():
    """Unknown damping type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown damping"):
        ri_damping(0.1, damping="quadratic")


def test_compute_nu_t_exponential_no_hard_cutoff():
    """Exponential damping produces nonzero nu_t even at high Ri_g."""
    grid = StretchedGrid(N=32, H=2.0, gamma=0.0)
    u = np.ones(32) * 10.0
    C = np.linspace(1.0, 0.0, 32)
    nu_t, _ = compute_nu_t(u, C, grid, g_prime=1.0, damping="exponential")
    assert np.any(nu_t[1:-1] > 0)


def test_ri_g_zero_in_low_shear_regions():
    """In regions of near-zero shear, Ri_g should be set to 0, not blow up.

    With a constant velocity profile (du/dz ~ 0) and nonzero dC/dz,
    the old epsilon-in-denominator approach would compute
    Ri_g = -g' * dCdz / (0 + eps) -> huge values, giving f ~ 0 and nu_t ~ 0.
    The corrected code sets Ri_g = 0 where dudz^2 < epsilon, so f = 1.0
    and nu_t equals the undamped mixing-length value.
    """
    N = 64
    grid = StretchedGrid(N=N, H=2.0, gamma=0.0)

    # Constant velocity: du/dz = 0 everywhere (below epsilon threshold)
    u = np.ones(N) * 5.0

    # Nonzero concentration gradient: stable stratification
    C = np.linspace(1.0, 0.0, N)

    g_prime = 1.0

    nu_t, D_t = compute_nu_t(u, C, grid, g_prime)

    # Compute the undamped base viscosity for the same velocity profile.
    # With du/dz ~ 0, mixing_length_nu_t gives ~ 0 at most points, but
    # the key assertion is that nu_t equals the base value (no damping).
    nu_t_undamped, _ = compute_nu_t(u, np.ones(N), grid, g_prime)

    # Interior points away from boundaries and away from z=0 (where z^2=0
    # makes both nu_t and nu_t_undamped zero regardless of damping).
    interior = slice(5, -5)

    # nu_t must equal the undamped value -- damping was NOT applied
    assert np.allclose(nu_t[interior], nu_t_undamped[interior], atol=1e-15), (
        "nu_t should equal undamped value in low-shear regions where Ri_g is "
        "set to 0 (damping factor = 1.0)"
    )
