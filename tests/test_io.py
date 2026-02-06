# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

import numpy as np
from fluidflow.io import save_run, load_run


def test_save_and_load_roundtrip(tmp_path):
    result = {
        "params": {"Re": 500, "S": 0.01, "Lambda": 1.0},
        "viscosity_ratio": 42.5,
        "drag_coefficient": 0.003,
        "kinetic_energy": 1.2,
        "regime": "turbulent",
    }
    path = tmp_path / "run_001.json"
    save_run(result, str(path))
    loaded = load_run(str(path))
    assert loaded["params"]["Re"] == 500
    assert loaded["viscosity_ratio"] == 42.5
    assert loaded["regime"] == "turbulent"
