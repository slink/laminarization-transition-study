# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT


import numpy as np

class Burgers1D:
    def __init__(self, nu, dx):
        self.nu = nu
        self.dx = dx

    def rhs(self, u, t):
        dudx = (np.roll(u, -1) - np.roll(u, 1)) / (2*self.dx)
        d2udx2 = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / self.dx**2
        return -u * dudx + self.nu * d2udx2
