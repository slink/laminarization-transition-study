# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

import numpy as np
from fluidflow.grid import StretchedGrid
from fluidflow.solvers.operators import diffusion_operator_matrix
from fluidflow.solvers.imex import IMEXEuler


def test_pure_diffusion_decays():
    """A Gaussian with Dirichlet BCs should diffuse and flatten over time.

    After 200 steps, the peak should be less than 50% of its initial value.
    """
    N = 64
    H = 1.0
    grid = StretchedGrid(N=N, H=H, gamma=0.0)  # uniform grid
    nu = np.full(N, 0.01)
    L = diffusion_operator_matrix(nu, grid)

    # Initial condition: Gaussian centred at H/2 with Dirichlet u=0 at both ends
    u = np.exp(-((grid.z - H / 2) ** 2) / (2 * 0.05**2))
    u[0] = 0.0
    u[-1] = 0.0
    peak_initial = u.max()

    dt = 0.5 * (grid.z[1] - grid.z[0]) ** 2 / 0.01  # moderate time step
    stepper = IMEXEuler(
        explicit_rhs=None,
        L_matrix=L,
        dt=dt,
        grid=grid,
        bc_type="dirichlet",
        bc_values=(0.0, 0.0),
    )

    t = 0.0
    for _ in range(200):
        u = stepper.step(u, t)
        t += dt

    # Peak must have decayed significantly
    assert u.max() < 0.5 * peak_initial, (
        f"Peak {u.max():.4f} did not decay below 50% of initial {peak_initial:.4f}"
    )


def test_steady_state_linear():
    """With u=0 at z=0 and u=1 at z=H, diffusion should preserve a linear profile.

    A linear profile is the steady state of pure diffusion with these
    Dirichlet BCs, so the solver should not alter it.
    """
    N = 64
    H = 1.0
    grid = StretchedGrid(N=N, H=H, gamma=0.0)  # uniform grid
    nu = np.full(N, 0.1)
    L = diffusion_operator_matrix(nu, grid)

    # Initial condition: exact linear profile
    u = grid.z / H

    dt = 0.5 * (grid.z[1] - grid.z[0]) ** 2 / 0.1
    stepper = IMEXEuler(
        explicit_rhs=None,
        L_matrix=L,
        dt=dt,
        grid=grid,
        bc_type="dirichlet",
        bc_values=(0.0, 1.0),
    )

    t = 0.0
    for _ in range(100):
        u = stepper.step(u, t)
        t += dt

    expected = grid.z / H
    assert np.allclose(u, expected, atol=1e-10), (
        f"Linear profile not preserved.  Max error: {np.max(np.abs(u - expected)):.2e}"
    )


def test_neumann_top_bc():
    """Zero-flux top BC (du/dz=0 at z=H) with Dirichlet u=0 at bottom.

    Start with u=1 everywhere except u[0]=0.  After stepping, the top
    boundary value u[-1] should remain well above zero because the
    Neumann condition prevents flux out of the top.
    """
    N = 64
    H = 1.0
    grid = StretchedGrid(N=N, H=H, gamma=0.0)
    nu = np.full(N, 0.01)
    L = diffusion_operator_matrix(nu, grid)

    # Initial condition: step function
    u = np.ones(N)
    u[0] = 0.0

    dt = 0.5 * (grid.z[1] - grid.z[0]) ** 2 / 0.01
    stepper = IMEXEuler(
        explicit_rhs=None,
        L_matrix=L,
        dt=dt,
        grid=grid,
        bc_type="neumann_top",
        bc_values=(0.0, None),
    )

    t = 0.0
    for _ in range(50):
        u = stepper.step(u, t)
        t += dt

    # The top value should remain positive thanks to the zero-flux BC
    assert u[-1] > 0.1, (
        f"Top boundary u[-1]={u[-1]:.4f} decayed too much; Neumann BC not working"
    )
