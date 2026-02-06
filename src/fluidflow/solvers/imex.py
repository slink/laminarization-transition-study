# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT


import numpy as np

from fluidflow.solvers.tridiagonal import thomas_solve


class IMEXEuler:
    """IMEX (Implicit-Explicit) Euler time integrator.

    Treats diffusion implicitly via a tridiagonal solve and any other
    right-hand-side terms explicitly:

        (I - dt*L) u^{n+1} = u^n + dt * f(u^n, t^n)

    where L is the diffusion operator matrix and f is the explicit RHS.

    Parameters
    ----------
    explicit_rhs : callable(u, t) -> array
        Explicit right-hand side function (e.g. advection).
        May be ``None`` for pure-diffusion problems.
    L_matrix : ndarray, shape (N, N)
        Diffusion operator from ``diffusion_operator_matrix``.
        Must be tridiagonal with zero boundary rows.
    dt : float
        Time step size.
    grid : StretchedGrid
        The computational grid.
    bc_type : str
        Boundary condition type. One of:
        - ``"dirichlet"``: Dirichlet conditions at both ends.
        - ``"neumann_top"``: Dirichlet at bottom (z=0), zero-flux
          Neumann (du/dz=0) at top (z=H).
    bc_values : tuple of float
        Boundary values ``(bottom, top)``.
        For ``"dirichlet"``: prescribed values at both boundaries.
        For ``"neumann_top"``: ``bottom`` is the Dirichlet value at z=0;
        ``top`` is ignored (the Neumann condition du/dz=0 is enforced).
    """

    def __init__(self, explicit_rhs, L_matrix, dt, grid, bc_type, bc_values):
        self.f = explicit_rhs
        self.dt = dt
        self.grid = grid
        self.bc_type = bc_type
        self.bc_values = bc_values

        N = grid.N
        L = L_matrix

        # Build (I - dt*L) and extract tridiagonal bands.
        # M = I - dt*L
        M = np.eye(N) - dt * L

        # Extract the three diagonals of M.
        # lower[i] = M[i, i-1]  for i = 1..N-1;  lower[0] is unused.
        # diag[i]  = M[i, i]    for i = 0..N-1
        # upper[i] = M[i, i+1]  for i = 0..N-2;  upper[-1] is unused.
        self._lower = np.zeros(N)
        self._diag = np.zeros(N)
        self._upper = np.zeros(N)

        self._diag[:] = np.diag(M, 0)
        self._lower[1:] = np.diag(M, -1)
        self._upper[:-1] = np.diag(M, 1)

        # Apply boundary conditions to the matrix rows.
        if bc_type == "dirichlet":
            # Row 0: u[0] = bc_values[0]  =>  1*u[0] = bc_values[0]
            self._lower[0] = 0.0
            self._diag[0] = 1.0
            self._upper[0] = 0.0
            # Row N-1: u[-1] = bc_values[1]  =>  1*u[-1] = bc_values[1]
            self._lower[-1] = 0.0
            self._diag[-1] = 1.0
            self._upper[-1] = 0.0

        elif bc_type == "neumann_top":
            # Row 0: Dirichlet at bottom  =>  1*u[0] = bc_values[0]
            self._lower[0] = 0.0
            self._diag[0] = 1.0
            self._upper[0] = 0.0
            # Row N-1: du/dz = 0 at top (free-slip)
            # First-order approximation u[-1]=u[-2]; sufficient because the top boundary
            # is placed far from the bed where the solution is nearly uniform.
            # A second-order 3-point stencil would break the tridiagonal structure.
            self._lower[-1] = -1.0
            self._diag[-1] = 1.0
            self._upper[-1] = 0.0

        else:
            raise ValueError(f"Unknown bc_type: {bc_type!r}")

    def step(self, u, t):
        """Advance solution by one time step.

        Parameters
        ----------
        u : ndarray, shape (N,)
            Current solution.
        t : float
            Current time.

        Returns
        -------
        u_new : ndarray, shape (N,)
            Solution at t + dt.
        """
        dt = self.dt

        # Explicit predictor: rhs = u + dt * f(u, t)
        if self.f is not None:
            rhs = u + dt * self.f(u, t)
        else:
            rhs = u.copy()

        # Apply boundary conditions to the RHS vector.
        if self.bc_type == "dirichlet":
            rhs[0] = self.bc_values[0]
            rhs[-1] = self.bc_values[1]
        elif self.bc_type == "neumann_top":
            rhs[0] = self.bc_values[0]
            rhs[-1] = 0.0  # u[-1] - u[-2] = 0

        # Solve tridiagonal system (I - dt*L) u^{n+1} = rhs.
        u_new = thomas_solve(
            self._lower.copy(),
            self._diag.copy(),
            self._upper.copy(),
            rhs,
        )

        return u_new
