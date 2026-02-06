# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

# tests/test_validation_diffusion.py
import numpy as np
from fluidflow.grid import StretchedGrid
from fluidflow.solvers.operators import diffusion_operator_matrix
from fluidflow.solvers.imex import IMEXEuler

def analytic_diffusion(z, t, nu, H, n_terms=50):
    """Analytic solution for diffusion of sin(pi*z/H) with Dirichlet BCs."""
    return np.sin(np.pi * z / H) * np.exp(-nu * (np.pi / H)**2 * t)

def test_diffusion_convergence():
    """Implicit diffusion should converge at 2nd order in space."""
    nu_val = 0.1
    H = 1.0
    T = 0.1
    errors = []

    for N in [32, 64, 128]:
        g = StretchedGrid(N=N, H=H, gamma=0.01)  # nearly uniform for clean convergence test
        nu = np.full(g.N, nu_val)
        L = diffusion_operator_matrix(nu, g)
        dt = 0.0001  # small enough that time error is negligible
        solver = IMEXEuler(
            explicit_rhs=lambda u, t: np.zeros_like(u),
            L_matrix=L, dt=dt, grid=g,
            bc_type="dirichlet", bc_values=(0.0, 0.0),
        )

        u = np.sin(np.pi * g.z / H)
        t = 0.0
        while t < T:
            u = solver.step(u, t)
            t += dt

        exact = analytic_diffusion(g.z, T, nu_val, H)
        errors.append(np.max(np.abs(u - exact)))

    # Check ~2nd order convergence
    ratio = errors[0] / errors[1]
    assert ratio > 3.0, f"Expected ~4x error reduction, got {ratio:.2f}"
