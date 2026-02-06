# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT


class RK4:
    def __init__(self, model, dt):
        self.model = model
        self.dt = dt

    def step(self, u, t):
        f = self.model.rhs
        dt = self.dt
        k1 = f(u, t)
        k2 = f(u + 0.5*dt*k1, t + 0.5*dt)
        k3 = f(u + 0.5*dt*k2, t + 0.5*dt)
        k4 = f(u + dt*k3, t + dt)
        return u + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
