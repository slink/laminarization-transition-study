# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT


import numpy as np
from fluidflow.models.burgers import Burgers1D
from fluidflow.solvers.time_integrators import RK4

def run(Re, nx=512, T=1.0):
    L = 2*np.pi
    dx = L / nx
    nu = 1/Re
    x = np.linspace(0, L, nx, endpoint=False)
    u = np.sin(x)
    model = Burgers1D(nu, dx)
    solver = RK4(model, dt=0.2*dx)

    t = 0.0
    while t < T:
        u = solver.step(u, t)
        t += solver.dt

    return np.max(np.abs(u))

if __name__ == "__main__":
    for Re in [50, 100, 200, 400]:
        print(Re, run(Re))
