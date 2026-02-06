# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

# tests/test_sweep.py
import numpy as np
from fluidflow.sweep import single_run, build_sweep_grid


def test_single_run_returns_result():
    """A single run should return a dict with expected keys."""
    params = dict(Re=200, S=0.01, Lambda=0.5, N=32, H=5.0, gamma=2.0,
                  Sc_t=1.0, n_cycles=3)
    result = single_run(params)
    assert "params" in result
    assert "viscosity_ratio" in result
    assert "regime" in result
    assert "converged" in result


def test_build_sweep_grid():
    """Sweep grid should produce correct number of parameter combos."""
    Re_vals = [100, 200]
    S_vals = [0.01, 0.1]
    Lambda_vals = [0.0, 1.0, 2.0]
    grid = build_sweep_grid(Re_vals, S_vals, Lambda_vals, N=32, H=5.0, gamma=2.0)
    assert len(grid) == 2 * 2 * 3  # 12 combos
    assert all("Re" in p for p in grid)


def test_single_run_zero_lambda():
    """With Lambda=0, should classify as turbulent (no stratification)."""
    params = dict(Re=500, S=0.01, Lambda=0.0, N=32, H=5.0, gamma=2.0,
                  Sc_t=1.0, n_cycles=5)
    result = single_run(params)
    assert result["regime"] == "turbulent"


def test_run_sweep_parallel():
    """run_sweep with max_workers=2 should produce same results as serial."""
    from fluidflow.sweep import run_sweep

    # Small 2-point grid: two Re values, one S and Lambda each
    param_list = build_sweep_grid(
        Re_vals=[50, 100],
        S_vals=[0.01],
        Lambda_vals=[0.5],
        N=32, H=5.0, gamma=2.0, Sc_t=1.0, n_cycles=2,
    )
    assert len(param_list) == 2

    # Run in parallel
    results = run_sweep(param_list, max_workers=2)

    assert len(results) == 2

    # Each result must contain expected keys
    expected_keys = {"params", "viscosity_ratio", "regime", "converged", "n_cycles_run"}
    for r in results:
        assert expected_keys.issubset(r.keys())
        assert r["regime"] in ("laminar", "turbulent")

    # Compare with serial single_run to verify parallel execution is faithful
    for params in param_list:
        serial_result = single_run(params)
        # Find matching parallel result by (Re, S, Lambda)
        matching = [
            r for r in results
            if r["params"]["Re"] == params["Re"]
            and r["params"]["S"] == params["S"]
            and r["params"]["Lambda"] == params["Lambda"]
        ]
        assert len(matching) == 1
        parallel_result = matching[0]
        # Regime and viscosity ratio must agree
        assert parallel_result["regime"] == serial_result["regime"]
        assert np.isclose(
            parallel_result["viscosity_ratio"],
            serial_result["viscosity_ratio"],
            rtol=1e-10,
        )
