# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

# tests/test_integration.py
import numpy as np
from fluidflow.sweep import single_run, build_sweep_grid
from fluidflow.io import save_run, load_run


def test_end_to_end_single_run():
    """Full pipeline: model -> step -> diagnostics -> result dict."""
    params = dict(Re=200, S=0.01, Lambda=0.5, N=32, H=5.0, gamma=2.0,
                  Sc_t=1.0, n_cycles=3)
    result = single_run(params)

    assert result["params"]["Re"] == 200
    assert result["regime"] in ("turbulent", "laminar")
    assert isinstance(result["viscosity_ratio"], float)
    assert isinstance(result["drag_coefficient"], float)
    assert isinstance(result["kinetic_energy"], float)


def test_end_to_end_save_load(tmp_path):
    """Full pipeline through save/load."""
    params = dict(Re=200, S=0.01, Lambda=0.0, N=32, H=5.0, gamma=2.0,
                  Sc_t=1.0, n_cycles=2)
    result = single_run(params)
    path = str(tmp_path / "test_run.json")
    save_run(result, path)
    loaded = load_run(path)
    assert loaded["regime"] == result["regime"]


def test_mini_sweep():
    """A 2x1x2 sweep should produce 4 results."""
    grid = build_sweep_grid(
        Re_vals=[100, 200],
        S_vals=[0.01],
        Lambda_vals=[0.0, 1.0],
        N=32, H=5.0, gamma=2.0, n_cycles=2,
    )
    assert len(grid) == 4
    # Run sequentially (no multiprocessing in test)
    results = [single_run(p) for p in grid]
    assert len(results) == 4
    assert all("regime" in r for r in results)
