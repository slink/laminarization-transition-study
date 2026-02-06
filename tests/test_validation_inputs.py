# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

import pytest
from fluidflow.grid import StretchedGrid
from fluidflow.models.oscillatory_bl import OscillatoryBLModel

def test_grid_rejects_small_N():
    with pytest.raises(ValueError):
        StretchedGrid(N=2, H=1.0, gamma=2.0)

def test_grid_rejects_nonpositive_H():
    with pytest.raises(ValueError):
        StretchedGrid(N=64, H=0.0, gamma=2.0)

def test_grid_rejects_negative_gamma():
    with pytest.raises(ValueError):
        StretchedGrid(N=64, H=1.0, gamma=-1.0)

def test_model_rejects_zero_Re():
    with pytest.raises(ValueError):
        OscillatoryBLModel({"Re": 0, "S": 1.0, "Lambda": 0.1})
