# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

# tests/test_sweep_utils.py
import csv
import json
import os

from fluidflow.sweep_utils import (
    save_sweep_results,
    load_phase_summary,
    configure_logging,
    print_summary_table,
)


def _make_results():
    """Create minimal sweep results for testing."""
    return [
        {
            "params": {"Re": 200.0, "S": 0.01, "Lambda": 0.5},
            "regime": "laminar",
            "converged": True,
            "viscosity_ratio": 3.456,
            "n_cycles_run": 15,
        },
        {
            "params": {"Re": 500.0, "S": 0.1, "Lambda": 0.0},
            "regime": "turbulent",
            "converged": False,
            "viscosity_ratio": 18.123,
            "n_cycles_run": 20,
        },
    ]


def test_save_and_load_summary_roundtrip(tmp_path):
    """save_sweep_results then load_phase_summary should roundtrip."""
    results = _make_results()
    summary_rows = save_sweep_results(results, str(tmp_path))

    # JSON files should exist
    assert os.path.exists(tmp_path / "Re200.0_S0.01_L0.5.json")
    assert os.path.exists(tmp_path / "Re500.0_S0.1_L0.0.json")

    # summary.csv should exist
    csv_path = tmp_path / "summary.csv"
    assert csv_path.exists()

    loaded = load_phase_summary(str(csv_path))
    assert len(loaded) == 2
    assert loaded[0]["Re"] == 200.0
    assert loaded[0]["S"] == 0.01
    assert loaded[0]["Lambda"] == 0.5
    assert loaded[0]["regime"] == "laminar"
    assert loaded[1]["Re"] == 500.0
    assert loaded[1]["regime"] == "turbulent"


def test_load_phase_summary_types(tmp_path):
    """Loaded summary should have proper types: float for numerics, bool for converged."""
    results = _make_results()
    save_sweep_results(results, str(tmp_path))

    csv_path = str(tmp_path / "summary.csv")
    loaded = load_phase_summary(csv_path)

    row = loaded[0]
    assert isinstance(row["Re"], float)
    assert isinstance(row["S"], float)
    assert isinstance(row["Lambda"], float)
    assert isinstance(row["viscosity_ratio"], float)
    assert isinstance(row["converged"], bool)
    assert row["converged"] is True

    row2 = loaded[1]
    assert row2["converged"] is False


def test_configure_logging(tmp_path):
    """configure_logging should create a log file in the output directory."""
    import logging

    logger = configure_logging(str(tmp_path), "test_phase")

    logger.info("test message")

    # Flush handlers
    for h in logger.handlers:
        h.flush()

    log_files = [f for f in os.listdir(tmp_path) if f.endswith(".log")]
    assert len(log_files) == 1
    assert "test_phase" in log_files[0]

    # Clean up handlers to avoid leaking between tests
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()


def test_print_summary_table(capsys):
    """print_summary_table should print formatted rows."""
    rows = [
        {"Re": 200.0, "S": 0.01, "Lambda": 0.5, "regime": "laminar",
         "converged": True, "viscosity_ratio": 3.456},
    ]
    print_summary_table(rows)
    captured = capsys.readouterr()
    assert "200.0" in captured.out
    assert "laminar" in captured.out
    assert "3.456" in captured.out
