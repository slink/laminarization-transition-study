# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

import numpy as np


class StretchedGrid:
    """1D stretched grid with tanh clustering near z=0 (bed).

    Computational coordinate xi in [0, 1] mapped to physical z in [0, H].
    z(xi) = H * [1 - tanh(gamma * (1 - xi)) / tanh(gamma)]

    When gamma -> 0, grid is uniform.

    Attributes:
        N: number of grid points
        H: domain height
        gamma: stretching parameter (higher = more clustering near bed)
        xi: uniform computational coordinate, shape (N,)
        z: physical coordinate, shape (N,)
        dz_dxi: grid metric dz/dxi, shape (N,)
        dxi_dz: inverse metric dxi/dz, shape (N,)
        dz: local grid spacing in physical space, shape (N,) (approx)
    """

    def __init__(self, N, H, gamma):
        if N < 3:
            raise ValueError(f"N must be >= 3 (FD stencils need at least 3 points), got {N}")
        if H <= 0:
            raise ValueError(f"H must be positive, got {H}")
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")

        self.N = N
        self.H = H
        self.gamma = gamma

        self.xi = np.linspace(0.0, 1.0, N)

        if abs(gamma) < 1e-10:
            self.z = H * self.xi
            self.dz_dxi = np.full(N, H)
        else:
            self.z = H * (1.0 - np.tanh(gamma * (1.0 - self.xi)) / np.tanh(gamma))
            self.dz_dxi = (
                H * gamma / np.tanh(gamma)
                * (1.0 / np.cosh(gamma * (1.0 - self.xi))) ** 2
            )

        self.dxi_dz = 1.0 / self.dz_dxi
        self.dz = np.gradient(self.z)
