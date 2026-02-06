# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

# tests/test_sweep_progress.py
import logging
from unittest.mock import patch

from fluidflow.sweep import build_sweep_grid, run_sweep


def _small_param_list():
    return build_sweep_grid(
        Re_vals=[50, 100],
        S_vals=[0.01],
        Lambda_vals=[0.5],
        N=32, H=5.0, gamma=2.0, Sc_t=1.0, n_cycles=2,
    )


def test_run_sweep_preserves_order():
    """Results must be in the same order as param_list."""
    param_list = _small_param_list()
    results = run_sweep(param_list, max_workers=2)
    assert len(results) == len(param_list)
    for params, result in zip(param_list, results):
        assert result["params"]["Re"] == params["Re"]
        assert result["params"]["S"] == params["S"]
        assert result["params"]["Lambda"] == params["Lambda"]


def test_run_sweep_progress_false():
    """progress=False should work without error."""
    param_list = _small_param_list()
    results = run_sweep(param_list, max_workers=1, progress=False)
    assert len(results) == len(param_list)


def test_run_sweep_without_tqdm():
    """If tqdm is not installed, run_sweep should still work."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "tqdm" or name == "tqdm.auto":
            raise ImportError("mocked: no tqdm")
        return real_import(name, *args, **kwargs)

    param_list = _small_param_list()
    with patch("builtins.__import__", side_effect=mock_import):
        results = run_sweep(param_list, max_workers=1, progress=True)
    assert len(results) == len(param_list)


def test_run_sweep_logs_completions(caplog):
    """run_sweep should log 'Starting sweep' and 'Sweep complete'."""
    param_list = _small_param_list()
    with caplog.at_level(logging.INFO, logger="fluidflow.sweep"):
        run_sweep(param_list, max_workers=1)
    messages = " ".join(caplog.messages)
    assert "Starting sweep" in messages
    assert "Sweep complete" in messages
