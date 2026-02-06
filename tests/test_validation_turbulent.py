# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

"""Turbulent validation against Jensen, Sumer & Fredsoe (1989).

Validates that the model produces physically reasonable turbulent friction
factors for clear-fluid oscillatory boundary layers by comparing against
experimental data from:

    Jensen, B.L., Sumer, B.M. & Fredsoe, J. (1989). "Turbulent oscillatory
    boundary layers at high Reynolds numbers." J. Fluid Mech., 206, 265-297.

Reference data digitized from Figure 11 of that paper.

Convention conversion:
    Jensen uses Re_a = U_0^2 / (omega * nu).
    We use Re_delta = U_0 * delta / nu, where delta = sqrt(2*nu/omega).
    Relation: Re_a = Re_delta^2 / 2.
"""

import os

import numpy as np
import pytest
from fluidflow.sweep import single_run

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_jensen_fw():
    """Load digitized friction factor data from Jensen et al. (1989)."""
    path = os.path.join(DATA_DIR, "jensen1989_fw.csv")
    return np.loadtxt(path, delimiter=",", skiprows=6,
                      usecols=(0, 1, 2))  # Re_a, fw, Re_delta


@pytest.mark.slow
def test_friction_factor_vs_jensen1989():
    """Model friction factor within 50% of Jensen et al. (1989) for clear-fluid OBL.

    We compare at the two lowest Reynolds numbers from Jensen (Re_delta ~ 394
    and ~803), which are within the range our 1D algebraic model can represent.
    The tolerance is generous (50%) because:
    - Jensen measured f_w from peak wall shear; we compute cycle-averaged c_f
      and scale by pi/2 to approximate the peak
    - Our algebraic mixing-length closure is simpler than DNS/experiment
    - The model is 1D with no turbulent kinetic energy transport

    The test verifies order-of-magnitude agreement, not quantitative precision.
    """
    ref = load_jensen_fw()

    # Only test the two lowest Re points (Re_delta ~ 394, ~803)
    # Higher Re requires finer grids and much longer runs
    test_points = ref[ref[:, 2] < 1000]

    for Re_a, fw_ref, Re_delta in test_points:
        Re = float(Re_delta)
        params = dict(
            Re=Re, S=0.0, Lambda=0.0,
            N=128, H=5.0, gamma=2.0,
            Sc_t=1.0, n_cycles=20,
        )
        result = single_run(params)

        # Our c_f is cycle-averaged |tau_bed| / (0.5 * U0^2).
        # Jensen's f_w uses max tau_bed over the cycle.
        # For sinusoidal forcing: <|sin|> = 2/pi, so f_w ~ (pi/2) * c_f.
        cf_model = result["drag_coefficient"]
        fw_model = (np.pi / 2) * cf_model

        rel_error = abs(fw_model - fw_ref) / fw_ref
        assert rel_error < 0.50, (
            f"Re_delta={Re:.0f}: model fw={fw_model:.4f}, "
            f"ref fw={fw_ref:.4f}, rel_error={rel_error:.2f}"
        )


@pytest.mark.slow
def test_clear_fluid_is_turbulent():
    """Clear-fluid OBL at Re=500 should classify as turbulent."""
    params = dict(
        Re=500, S=0.0, Lambda=0.0,
        N=128, H=5.0, gamma=2.0,
        Sc_t=1.0, n_cycles=10,
    )
    result = single_run(params)
    assert result["regime"] == "turbulent", (
        f"Clear-fluid at Re=500 should be turbulent, "
        f"got regime={result['regime']}, "
        f"viscosity_ratio={result['viscosity_ratio']:.1f}"
    )
