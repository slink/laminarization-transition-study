#!/usr/bin/env python3
# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

"""Run revised Phase 4 production simulations.

Reads parameter points from results/phase4_revised_params.json and runs
high-resolution (N=256, 40 cycles) simulations at points that straddle
the laminar transition boundaries identified in Phase 3.

Usage:
    python scripts/run_phase4_revised.py [--workers N]
"""

import argparse
import json
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fluidflow.sweep import run_sweep
from fluidflow.sweep_utils import save_sweep_results, print_summary_table, configure_logging


def load_revised_params(json_path):
    """Load revised parameter points from JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    return data["runs"]


def build_param_list(runs, N=256, H=5.0, gamma=2.0, Sc_t=1.0, n_cycles=40):
    """Convert JSON run specs to full parameter dicts for single_run."""
    param_list = []
    for run in runs:
        param_list.append({
            "Re": float(run["Re"]),
            "S": float(run["S"]),
            "Lambda": float(run["Lambda"]),
            "N": N,
            "H": H,
            "gamma": gamma,
            "Sc_t": Sc_t,
            "n_cycles": n_cycles,
            "damping": "linear",
        })
    return param_list


def main():
    parser = argparse.ArgumentParser(
        description="Run revised Phase 4 production simulations"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Max parallel workers (default: cpu count)"
    )
    parser.add_argument(
        "--params", type=str,
        default="results/phase4_revised_params.json",
        help="Path to revised parameters JSON"
    )
    parser.add_argument(
        "--outdir", type=str,
        default="results/phase4-production-revised",
        help="Output directory"
    )
    args = parser.parse_args()

    # Setup logging
    logger = configure_logging(args.outdir, "phase4-revised")

    # Load revised parameters
    runs = load_revised_params(args.params)
    param_list = build_param_list(runs, N=256, n_cycles=40)

    n_cases = len(param_list)
    print(f"Phase 4 Revised: {n_cases} cases")
    print(f"Grid: N=256, H=5.0, gamma=2.0, Sc_t=1.0")
    print(f"Cycles: 40, Workers: {args.workers or 'auto'}")
    print()

    # Show expected regime distribution
    expected_L = sum(1 for r in runs if r["expected"] == "L")
    expected_T = sum(1 for r in runs if r["expected"] == "T")
    print(f"Expected: {expected_L} laminar, {expected_T} turbulent")
    print()

    # Run simulations
    results = run_sweep(param_list, max_workers=args.workers)

    # Save results
    summary_rows = save_sweep_results(results, args.outdir)
    print_summary_table(summary_rows)

    # Compare actual vs expected
    actual_L = sum(1 for r in results if r["regime"] == "laminar")
    actual_T = sum(1 for r in results if r["regime"] == "turbulent")
    converged = sum(1 for r in results if r["converged"])

    print()
    print(f"Actual: {actual_L} laminar, {actual_T} turbulent")
    print(f"Converged: {converged}/{n_cases} ({100*converged/n_cases:.0f}%)")
    print(f"\nResults saved to {args.outdir}/")


if __name__ == "__main__":
    main()
