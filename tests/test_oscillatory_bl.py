# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
from fluidflow.models.oscillatory_bl import OscillatoryBLModel


def test_model_initialization():
    """Model should initialize with given parameters."""
    params = dict(Re=500, S=0.01, Lambda=1.0, N=64, H=5.0, gamma=2.0, Sc_t=1.0)
    model = OscillatoryBLModel(params)
    assert model.grid.N == 64
    assert model.grid.H == 5.0


def test_one_step_runs():
    """A single timestep should not crash and should return correct shapes."""
    params = dict(Re=500, S=0.01, Lambda=1.0, N=64, H=5.0, gamma=2.0, Sc_t=1.0)
    model = OscillatoryBLModel(params)
    u, C = model.get_initial_condition()
    assert u.shape == (64,)
    assert C.shape == (64,)
    u_new, C_new = model.step(u, C, t=0.0)
    assert u_new.shape == (64,)
    assert C_new.shape == (64,)


def test_no_slip_bc():
    """Velocity at bed should be zero after stepping."""
    params = dict(Re=500, S=0.01, Lambda=1.0, N=64, H=5.0, gamma=2.0, Sc_t=1.0)
    model = OscillatoryBLModel(params)
    u, C = model.get_initial_condition()
    t = 0.0
    for _ in range(10):
        u, C = model.step(u, C, t)
        t += model.dt
    assert np.isclose(u[0], 0.0)


def test_concentration_bc():
    """Concentration at bed should be C_ref."""
    params = dict(Re=500, S=0.01, Lambda=1.0, N=64, H=5.0, gamma=2.0, Sc_t=1.0)
    model = OscillatoryBLModel(params)
    u, C = model.get_initial_condition()
    t = 0.0
    for _ in range(10):
        u, C = model.step(u, C, t)
        t += model.dt
    assert np.isclose(C[0], model.C_ref)


def test_concentration_non_negative():
    """Concentration should remain non-negative."""
    params = dict(Re=200, S=0.05, Lambda=2.0, N=64, H=5.0, gamma=2.0, Sc_t=1.0)
    model = OscillatoryBLModel(params)
    u, C = model.get_initial_condition()
    t = 0.0
    for _ in range(200):
        u, C = model.step(u, C, t)
        t += model.dt
    assert np.all(C >= -1e-10)


def test_oscillatory_forcing_active():
    """Forcing F0*sin(omega*t) should produce nonzero velocity when t ~ pi/2.

    At t = pi/2 with omega = 1, sin(omega*t) = 1 so forcing is at its
    positive peak.  Stepping to that time and checking that interior
    velocities are positive validates that the oscillatory forcing is
    actually driving the flow.
    """
    params = dict(Re=500, S=0.01, Lambda=1.0, N=64, H=5.0, gamma=2.0, Sc_t=1.0)
    model = OscillatoryBLModel(params)
    u, C = model.get_initial_condition()

    # Step until t reaches approximately pi/2 (quarter period)
    target_t = np.pi / 2.0
    n_steps = int(np.ceil(target_t / model.dt))
    t = 0.0
    for _ in range(n_steps):
        u, C = model.step(u, C, t)
        t += model.dt

    # Interior velocity (excluding no-slip bed node) should be nonzero
    # and predominantly positive because the forcing integral is positive
    # over [0, pi/2].
    u_interior = u[1:]
    assert np.any(np.abs(u_interior) > 1e-8), (
        "Interior velocity is effectively zero -- oscillatory forcing not active"
    )
    assert np.mean(u_interior) > 0.0, (
        "Mean interior velocity should be positive after positive forcing phase"
    )


def test_turbulence_produces_nonzero_nu_t():
    """With turbulence_enabled=True and moderate Re, nu_t should be nonzero."""
    from fluidflow.models.closures import compute_nu_t

    params = {"Re": 200, "S": 1.0, "Lambda": 0.1, "N": 32, "H": 1.0, "gamma": 2.0,
              "turbulence_enabled": True}
    model = OscillatoryBLModel(params)
    g = model.grid

    # Initialize and step a few times to develop flow
    u = np.zeros(g.N)
    C = np.ones(g.N) * 0.5  # nonzero concentration
    t = 0.0
    for _ in range(50):
        u, C = model.step(u, C, t)
        t += model.dt

    # Compute nu_t from current state
    nu_t, D_t = compute_nu_t(u, C, g, g_prime=model.g_prime, Sc_t=model.Sc_t)
    assert np.any(nu_t > 0), "nu_t should be nonzero with turbulence enabled"


def test_stratification_increases_damping():
    """Increasing Lambda should reduce turbulent viscosity."""
    from fluidflow.models.closures import compute_nu_t

    nu_t_results = []
    for Lambda in [0.01, 1.0]:
        params = {"Re": 200, "S": 1.0, "Lambda": Lambda, "N": 32, "H": 1.0, "gamma": 2.0,
                  "turbulence_enabled": True}
        model = OscillatoryBLModel(params)
        g = model.grid
        u = np.zeros(g.N)
        C = np.ones(g.N) * 0.5
        t = 0.0
        for _ in range(50):
            u, C = model.step(u, C, t)
            t += model.dt
        nu_t, _ = compute_nu_t(u, C, g, g_prime=model.g_prime, Sc_t=model.Sc_t)
        nu_t_results.append(np.mean(nu_t))

    # Higher Lambda should give lower nu_t (more damping)
    assert nu_t_results[1] < nu_t_results[0]


def test_sediment_conservation():
    """Total sediment should not change dramatically with no-flux BCs."""
    params = {"Re": 100, "S": 0.5, "Lambda": 0.1, "N": 64, "H": 1.0, "gamma": 2.0,
              "turbulence_enabled": False}
    model = OscillatoryBLModel(params)
    g = model.grid
    u = np.zeros(g.N)
    C = np.exp(-g.z)  # Initial profile decaying with height

    initial_mass = np.trapezoid(C, g.z)

    t = 0.0
    for _ in range(100):
        u, C = model.step(u, C, t)
        t += model.dt

    final_mass = np.trapezoid(C, g.z)

    # With settling + diffusion and no-flux top BC + Dirichlet bottom BC,
    # mass may change, but shouldn't blow up or go negative
    assert final_mass > 0, "Sediment mass should remain positive"
    assert final_mass < initial_mass * 10, "Sediment mass should not blow up"


def test_model_accepts_damping_param():
    """OscillatoryBLModel stores and uses the damping parameter."""
    params = dict(Re=200, S=0.01, Lambda=0.1, N=32, H=5.0, gamma=2.0,
                  Sc_t=1.0, n_cycles=1, damping="exponential")
    model = OscillatoryBLModel(params)
    assert model.damping == "exponential"


def test_model_damping_defaults_to_linear():
    """Without explicit damping param, model defaults to linear."""
    params = dict(Re=200, S=0.01, Lambda=0.1, N=32, H=5.0, gamma=2.0)
    model = OscillatoryBLModel(params)
    assert model.damping == "linear"


def test_settling_moves_sediment_toward_bed():
    """With turbulence off, settling should increase C near bed relative to top.

    An initially uniform concentration profile with settling active (S > 0)
    and no turbulent diffusion should develop higher concentration near the
    bed (z=0) over time because w_s < 0 transports sediment downward.
    This test would have caught the original sign error in the settling term.
    """
    params = dict(Re=100, S=0.5, Lambda=0.0, N=64, H=5.0, gamma=2.0,
                  turbulence_enabled=False)
    model = OscillatoryBLModel(params)
    g = model.grid

    u = np.zeros(g.N)
    C = np.ones(g.N) * 0.5  # uniform initial concentration (except bed BC)
    C[0] = model.C_ref

    t = 0.0
    for _ in range(200):
        u, C = model.step(u, C, t)
        t += model.dt

    # Near-bed concentration should be higher than far-from-bed
    near_bed = np.mean(C[1:g.N // 4])
    far_bed = np.mean(C[3 * g.N // 4:])
    assert near_bed > far_bed, (
        f"Settling should move sediment toward bed: near_bed={near_bed:.4f}, "
        f"far_bed={far_bed:.4f}"
    )
