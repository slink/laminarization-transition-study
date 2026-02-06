# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

import numpy as np
from fluidflow.grid import StretchedGrid
from fluidflow.solvers.operators import ddz, diffusion_operator_bands
from fluidflow.solvers.tridiagonal import thomas_solve
from fluidflow.models.closures import compute_nu_t


class OscillatoryBLModel:
    """Coupled momentum + sediment oscillatory boundary layer.

    Nondimensional equations:
        u_t = F0*sin(omega*t) + d/dz[(1/Re + nu_t) * du/dz]
        C_t = -d/dz(w_s * C) + d/dz[D_t * dC/dz]    (w_s = -S)

    where nu_t is damped by stratification via Ri_g.

    Time integration: IMEX Euler.
        - Implicit: diffusion terms (tridiagonal solve)
        - Explicit: forcing, settling flux
        - Lagged: nu_t from current step
    """

    def __init__(self, params):
        if params["Re"] <= 0:
            raise ValueError(f"Re must be positive (prevents division by zero in nu = 1/Re), got {params['Re']}")
        if params.get("N", 0) < 3:
            raise ValueError(f"N must be >= 3 (FD stencils need at least 3 points), got {params.get('N')}")

        self.params = params
        self.Re = params["Re"]
        self.S = params["S"]
        self.Lambda = params["Lambda"]
        self.Sc_t = params.get("Sc_t", 1.0)
        self.damping = params.get("damping", "linear")
        self.turbulence_enabled = params.get("turbulence_enabled", True)

        self.grid = StretchedGrid(N=params["N"], H=params["H"], gamma=params["gamma"])
        self.nu = 1.0 / self.Re  # molecular viscosity (nondimensional)
        self.omega = 1.0  # nondimensional frequency
        self.F0 = 1.0     # nondimensional forcing amplitude
        self.C_ref = 1.0  # reference concentration at bed (nondimensional)

        # g' = Lambda in nondimensional form
        self.g_prime = self.Lambda

        # CFL-based timestep
        dz_min = np.min(np.diff(self.grid.z))
        self.dt = 0.4 * dz_min / max(self.S, 1.0)

    def get_initial_condition(self):
        """Stokes layer IC for velocity, zero sediment."""
        g = self.grid
        u = np.exp(-g.z) * np.sin(g.z)
        u[0] = 0.0  # no-slip
        C = np.zeros(g.N)
        C[0] = self.C_ref
        return u, C

    def step(self, u, C, t):
        """Advance one IMEX Euler timestep.

        Returns: (u_new, C_new)
        """
        g = self.grid
        dt = self.dt
        N = g.N
        z = g.z

        # 1. Compute turbulent viscosity (lagged from current state)
        if self.turbulence_enabled:
            nu_t, D_t = compute_nu_t(u, C, g, g_prime=self.g_prime, Sc_t=self.Sc_t,
                                     damping=self.damping)
        else:
            nu_t = np.zeros(N)
            D_t = np.zeros(N)

        # 2. Momentum: (I - dt*L_nu) u^{n+1} = u^n + dt*F0*sin(omega*t)
        nu_eff = self.nu + nu_t
        L_lower, L_diag, L_upper = diffusion_operator_bands(nu_eff, z, N)

        # Build tridiagonal coefficients for (I - dt*L): vectorized
        lower_u = -dt * L_lower
        diag_u = 1.0 - dt * L_diag
        upper_u = -dt * L_upper

        rhs_u = u + dt * self.F0 * np.sin(self.omega * t)
        rhs_u[0] = 0.0   # no-slip
        # Free-slip top: du/dz = 0 at z[-1].
        # 1st-order one-sided: u[-1] = u[-2].
        # A 2nd-order 3-point backward stencil (h2^2*f[-1] + (h1^2-h2^2)*f[-2]
        # - h1^2*f[-3] = 0) would require coupling to index N-3, which exceeds
        # the tridiagonal bandwidth.  Because the stretched grid places the top
        # boundary far from the bed where gradients are negligible, the
        # first-order approximation is adequate.
        diag_u[0] = 1.0
        lower_u[0] = 0.0
        upper_u[0] = 0.0
        diag_u[-1] = 1.0
        lower_u[-1] = -1.0
        upper_u[-1] = 0.0
        rhs_u[-1] = 0.0

        u_new = thomas_solve(lower_u, diag_u, upper_u, rhs_u)

        # 3. Sediment: (I - dt*L_D) C^{n+1} = C^n - dt*d_z(w_s*C^n)
        D_eff = np.maximum(D_t, 1e-12)  # floor to avoid zero diffusivity
        L_lower_c, L_diag_c, L_upper_c = diffusion_operator_bands(D_eff, z, N)

        # Build tridiagonal coefficients: vectorized
        lower_c = -dt * L_lower_c
        diag_c = 1.0 - dt * L_diag_c
        upper_c = -dt * L_upper_c

        # Settling flux: upwind for w_s < 0 (downward advection in -z).
        # Information travels upward (toward +z / i+1), so bias the
        # one-sided difference toward i+1 (the upwind direction).
        settling = np.zeros(N)
        dz_fwd = z[2:] - z[1:-1]
        settling[1:-1] = -self.S * (C[2:] - C[1:-1]) / dz_fwd

        rhs_c = C - dt * settling
        rhs_c[0] = self.C_ref          # Dirichlet at bed
        # Zero-flux top: dC/dz = 0 at z[-1].
        # Same reasoning as momentum: 2nd-order backward stencil needs the
        # sub-sub-diagonal (index N-3) which breaks tridiagonal structure.
        # First-order is acceptable here; see momentum BC comment above.
        diag_c[0] = 1.0
        lower_c[0] = 0.0
        upper_c[0] = 0.0
        diag_c[-1] = 1.0
        lower_c[-1] = -1.0
        upper_c[-1] = 0.0
        rhs_c[-1] = 0.0

        C_new = thomas_solve(lower_c, diag_c, upper_c, rhs_c)

        # Clamp concentration to non-negative
        C_new = np.maximum(C_new, 0.0)

        return u_new, C_new
