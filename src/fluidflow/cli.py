# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

"""Command-line interface for fluidflow parameter sweeps."""

import argparse
import os
import sys

from fluidflow.sweep import build_sweep_grid, run_sweep
from fluidflow.sweep_utils import save_sweep_results, print_summary_table


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="fluidflow-sweep",
        description="Run laminarization parameter sweeps over (Re, S, Lambda).",
    )
    parser.add_argument(
        "--Re", nargs="+", type=float, required=True,
        help="Reynolds number values",
    )
    parser.add_argument(
        "--S", nargs="+", type=float, required=True,
        help="Settling number values",
    )
    parser.add_argument(
        "--Lambda", nargs="+", type=float, required=True,
        help="Stratification parameter values",
    )
    parser.add_argument(
        "-N", type=int, default=128,
        help="Number of grid points (default: 128)",
    )
    parser.add_argument(
        "-H", type=float, default=5.0,
        help="Domain height in Stokes-layer thicknesses (default: 5.0)",
    )
    parser.add_argument(
        "--gamma", type=float, default=2.0,
        help="Grid stretching parameter (default: 2.0)",
    )
    parser.add_argument(
        "--Sc_t", type=float, default=1.0,
        help="Turbulent Schmidt number (default: 1.0)",
    )
    parser.add_argument(
        "--cycles", type=int, default=20,
        help="Number of oscillation cycles (default: 20)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Max parallel workers (default: cpu count)",
    )
    parser.add_argument(
        "--damping", type=str, default="linear",
        choices=["linear", "exponential"],
        help="Richardson damping function type (default: linear)",
    )
    parser.add_argument(
        "--outdir", type=str, default="results",
        help="Output directory (default: results/)",
    )

    args = parser.parse_args(argv)

    param_list = build_sweep_grid(
        Re_vals=args.Re,
        S_vals=args.S,
        Lambda_vals=args.Lambda,
        N=args.N,
        H=args.H,
        gamma=args.gamma,
        Sc_t=args.Sc_t,
        n_cycles=args.cycles,
        damping=args.damping,
    )

    n_cases = len(param_list)
    print(f"Running {n_cases} cases (Re={args.Re}, S={args.S}, Lambda={args.Lambda})")
    print(f"Grid: N={args.N}, H={args.H}, gamma={args.gamma}, Sc_t={args.Sc_t}")
    print(f"Cycles: {args.cycles}, Workers: {args.workers or 'auto'}")
    print()

    results = run_sweep(param_list, max_workers=args.workers)

    summary_rows = save_sweep_results(results, args.outdir)
    print_summary_table(summary_rows)
    print(f"\nResults saved to {args.outdir}/")
    print(f"Summary: {os.path.join(args.outdir, 'summary.csv')}")
