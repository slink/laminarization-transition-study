# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

# src/fluidflow/diagnostics.py
import numpy as np
from fluidflow.solvers.operators import ddz


class CycleDiagnostics:
    """Accumulate and compute cycle-averaged diagnostics.

    Usage:
        diag = CycleDiagnostics(grid)
        for each timestep in cycle:
            diag.accumulate(u, C, nu_t, D_t, nu, t)
        result = diag.finalize()
    """

    def __init__(self, grid, threshold=10.0, U0=1.0, g_prime=0.0):
        self.grid = grid
        self.threshold = threshold
        self.U0 = U0
        self.g_prime = g_prime
        self.samples = []

    def accumulate(self, u, C, nu_t, D_t, nu, t):
        """Store one snapshot for cycle averaging."""
        g = self.grid
        dudz = ddz(u, g)

        self.samples.append({
            "u": u.copy(),
            "C": C.copy(),
            "nu_t": nu_t.copy(),
            "D_t": D_t.copy(),
            "dudz": dudz.copy(),
            "nu": nu,
            "t": t,
        })

    def finalize(self):
        """Compute cycle-averaged diagnostics from accumulated snapshots."""
        g = self.grid
        n = len(self.samples)

        # Cycle-average arrays
        nu_t_avg = np.mean([s["nu_t"] for s in self.samples], axis=0)
        dudz_avg = np.mean([s["dudz"] for s in self.samples], axis=0)
        C_avg = np.mean([s["C"] for s in self.samples], axis=0)
        D_t_avg = np.mean([s["D_t"] for s in self.samples], axis=0)
        nu = self.samples[0]["nu"]

        # Effective viscosity ratio (domain-averaged)
        viscosity_ratio = np.trapezoid(nu_t_avg, g.z) / g.z[-1] / nu

        # Reynolds stress profile: -nu_t * du/dz (cycle-averaged)
        reynolds_stress = -np.mean(
            [s["nu_t"] * s["dudz"] for s in self.samples], axis=0
        )

        # c_f = tau_bed / (0.5 * U0^2); U0=1 by default (nondimensional)
        tau_bed_samples = [(s["nu"] + s["nu_t"][0]) * np.abs(s["dudz"][0])
                          for s in self.samples]
        tau_bed = np.mean(tau_bed_samples)
        drag_coefficient = tau_bed / (0.5 * self.U0**2)

        # Kinetic energy: cycle-averaged integral of 0.5 * u^2 dz
        ke_samples = [0.5 * np.trapezoid(s["u"]**2, g.z) for s in self.samples]
        kinetic_energy = np.mean(ke_samples)

        # Sediment flux: diffusive flux part
        dCdz_avg = np.mean([ddz(s["C"], g) for s in self.samples], axis=0)
        sediment_flux = np.trapezoid(D_t_avg * np.abs(dCdz_avg), g.z)

        # Phase portrait: (u_bed, tau_bed) pairs
        phase_portrait = [
            (s["u"][1], (s["nu"] + s["nu_t"][0]) * np.abs(s["dudz"][0]))
            for s in self.samples
        ]

        # Classification
        regime = "turbulent" if viscosity_ratio > self.threshold else "laminar"

        # Cycle-averaged Ri_g using rms shear: <dC/dz> / <(du/dz)^2>
        # Uses cycle-mean of dudz^2 (always positive, no cancellation)
        # to avoid both sign-cancellation and flow-reversal singularities.
        epsilon = 1e-10
        dudz_sq_avg = np.mean([s["dudz"]**2 for s in self.samples], axis=0)
        Ri_g_avg = -self.g_prime * dCdz_avg / (dudz_sq_avg + epsilon)

        return {
            "viscosity_ratio": viscosity_ratio,
            "reynolds_stress": reynolds_stress,
            "drag_coefficient": drag_coefficient,
            "kinetic_energy": kinetic_energy,
            "sediment_flux": sediment_flux,
            "phase_portrait": phase_portrait,
            "regime": regime,
            "profiles": {
                "nu_t": nu_t_avg.tolist(),
                "C": C_avg.tolist(),
                "Ri_g": Ri_g_avg.tolist(),
                "z": g.z.tolist(),
            },
        }
