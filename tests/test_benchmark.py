# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

"""Smoke tests for the benchmark module."""

import pytest


@pytest.mark.slow
def test_run_all_benchmarks_smoke():
    """Smoke test: run_all_benchmarks returns expected keys with positive timings."""
    from fluidflow.benchmark import run_all_benchmarks

    results = run_all_benchmarks(N=16, verbose=False)

    # Micro-benchmark keys
    micro_keys = {
        "thomas_solve",
        "ddz",
        "diffusion_matrix",
        "compute_nu_t",
        "model_step",
    }
    for key in micro_keys:
        assert key in results, f"Missing micro-benchmark key: {key}"
        assert results[key]["median_ms"] > 0, f"{key} median_ms should be positive"

    # diffusion_bands should be present (added via the try/except in run_all_benchmarks)
    assert "diffusion_bands" in results, "Missing diffusion_bands benchmark"
    assert results["diffusion_bands"]["median_ms"] > 0

    # Macro-benchmark key
    assert "single_run" in results, "Missing single_run macro-benchmark"
    assert results["single_run"]["elapsed_s"] > 0, "single_run elapsed_s should be positive"
