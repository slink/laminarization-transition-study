# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

import numpy as np
from fluidflow.grid import StretchedGrid

def test_grid_endpoints():
    """Grid should span [0, H]."""
    g = StretchedGrid(N=64, H=1.0, gamma=2.0)
    assert g.z[0] == 0.0
    assert np.isclose(g.z[-1], 1.0)

def test_grid_clustering():
    """First cell should be smaller than last cell."""
    g = StretchedGrid(N=64, H=1.0, gamma=2.0)
    dz_first = g.z[1] - g.z[0]
    dz_last = g.z[-1] - g.z[-2]
    assert dz_first < dz_last

def test_metric_integrates_to_H():
    """Integral of dz/dxi over [0,1] should equal H."""
    g = StretchedGrid(N=256, H=5.0, gamma=2.0)
    dxi = 1.0 / (g.N - 1)
    integral = np.trapezoid(g.dz_dxi, dx=dxi)
    assert np.isclose(integral, 5.0, rtol=1e-3)

def test_uniform_grid_when_gamma_zero():
    """gamma=0 (or very small) should give approximately uniform spacing."""
    g = StretchedGrid(N=64, H=1.0, gamma=0.01)
    dz = np.diff(g.z)
    assert np.allclose(dz, dz[0], rtol=0.05)
