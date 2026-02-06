#!/usr/bin/env python3
# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

"""Re-entrant transition window expansion at N=256.

Runs ~134 new cases with finer Lambda spacing near the re-entrant transition
window at S=0.005 and S=0.01. The re-entrant transition is the most novel
finding in this research: at S=0.005, Re=1000, flow goes turbulent->laminar->
turbulent as Lambda increases. This script densifies coverage at transition
edges and at (Re,S) pairs where the re-entrant window's existence is uncertain.

Deduplicates against:
- results/archive/phase3-n256/   (archived Phase 3 N=256 data)
- results/phase1-n256/           (Phase 1 N=256 data, if present)
- results/reentrant-expansion/   (own output, for resume)

Features:
- Incremental saving: results saved as each case completes
- Resume capability: skips already-completed cases on restart
- Parallel execution with progress tracking
- Battery-aware pause/resume (macOS only): pauses when battery low, resumes when charged

Usage:
    python scripts/run_reentrant_expansion.py [--workers N] [--battery-pause PCT]

Examples:
    # Run all re-entrant expansion cases
    python scripts/run_reentrant_expansion.py

    # Run with 4 workers
    python scripts/run_reentrant_expansion.py --workers 4

    # Resume after interruption (just run the same command)
    python scripts/run_reentrant_expansion.py

    # Disable battery monitoring
    python scripts/run_reentrant_expansion.py --battery-pause 0
"""

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fluidflow.sweep import single_run
from fluidflow.io import save_run
from fluidflow.sweep_utils import configure_logging

# --- Re-entrant transition expansion parameter grid ---
# Keys: (Re, S) tuples; Values: list of Lambda values to probe
REENTRANT_CASES = {
    # S=0.005: refine transition edges + fill gaps at Re=500,750
    (300,  0.005): [0.028, 0.029, 0.030, 0.031, 0.032,
                    0.65, 0.66, 0.67, 0.68, 0.69, 0.70],
    (500,  0.005): np.linspace(0.1, 5.0, 20).tolist(),
    (750,  0.005): np.linspace(0.1, 5.0, 20).tolist(),
    (1000, 0.005): [0.26, 0.27, 0.28,
                    2.5, 2.6, 2.7, 2.8,
                    6.0, 7.0, 8.0, 10.0],

    # S=0.01: check if laminar band exists at higher Re
    (300,  0.01):  [0.055, 0.058, 0.060, 0.062, 0.065,
                    0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16],
    (500,  0.01):  np.linspace(0.05, 3.0, 20).tolist(),
    (750,  0.01):  np.linspace(0.05, 4.0, 20).tolist(),
    (1000, 0.01):  np.linspace(0.05, 4.0, 20).tolist(),
}

# Directories to deduplicate against (in addition to own output)
DEDUP_DIRS = [
    "results/archive/phase3-n256",
    "results/phase1-n256",
]


def get_battery_status():
    """Get battery percentage and charging status on macOS.

    Returns:
        (percentage, is_charging) or (None, None) if no battery/not macOS
    """
    try:
        result = subprocess.run(
            ["pmset", "-g", "batt"],
            capture_output=True, text=True, timeout=5
        )
        output = result.stdout

        # Parse actual percentage from battery line
        match = re.search(r'(\d+)%', output)
        if not match:
            return (None, None)  # No battery found

        percentage = int(match.group(1))

        # Detect charging: on AC power, or explicitly "charging" (not "discharging")
        on_ac = "AC Power" in output
        is_charging = on_ac or (
            "charging" in output.lower() and "discharging" not in output.lower()
        )

        return (percentage, is_charging)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return (None, None)  # Not macOS or pmset unavailable


def wait_for_battery(min_percent=80, check_interval=60):
    """Wait until battery percentage is above threshold.

    Args:
        min_percent: Resume when battery reaches this level
        check_interval: Seconds between checks
    """
    while True:
        pct, charging = get_battery_status()

        if pct is None:  # No battery, continue
            return
        if pct >= min_percent:
            status = "charging" if charging else "on battery"
            print(f"\n  Battery at {pct}% ({status}), resuming...")
            return

        status = "charging" if charging else "discharging"
        print(f"\r  Paused - battery at {pct}% ({status}), waiting for {min_percent}%...",
              end="", flush=True)
        time.sleep(check_interval)


def make_filename(Re, S, Lambda):
    """Generate consistent filename for a parameter set."""
    return f"Re{Re}_S{S}_L{Lambda}.json"


def _round_key(Re, S, Lambda, decimals=6):
    """Round a (Re, S, Lambda) tuple for approximate float comparison."""
    return (round(float(Re), decimals), round(float(S), decimals), round(float(Lambda), decimals))


def load_completed_cases(outdir):
    """Load set of rounded (Re, S, Lambda) tuples that are already completed."""
    completed = set()
    outdir = Path(outdir)
    if not outdir.exists():
        return completed

    for json_file in outdir.glob("Re*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            p = data["params"]
            completed.add(_round_key(p["Re"], p["S"], p["Lambda"]))
        except (json.JSONDecodeError, KeyError):
            # Skip malformed files
            continue

    return completed


def build_reentrant_params():
    """Build re-entrant expansion parameter list as (Re, S, Lambda) tuples."""
    params = []
    for (Re, S), lambdas in sorted(REENTRANT_CASES.items()):
        for lam in lambdas:
            params.append((float(Re), float(S), float(lam)))
    return params


def build_param_dict(Re, S, Lambda, N=256, H=5.0, gamma=2.0, Sc_t=1.0, n_cycles=20):
    """Build full parameter dict for single_run."""
    return {
        "Re": Re,
        "S": S,
        "Lambda": Lambda,
        "N": N,
        "H": H,
        "gamma": gamma,
        "Sc_t": Sc_t,
        "n_cycles": n_cycles,
        "damping": "linear",
    }


def update_summary_csv(outdir):
    """Rebuild summary.csv from all JSON files in outdir."""
    outdir = Path(outdir)
    rows = []

    for json_file in sorted(outdir.glob("Re*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
            p = data["params"]
            rows.append({
                "Re": p["Re"],
                "S": p["S"],
                "Lambda": p["Lambda"],
                "regime": data["regime"],
                "converged": data["converged"],
                "viscosity_ratio": data["viscosity_ratio"],
            })
        except (json.JSONDecodeError, KeyError):
            continue

    # Sort by Re, S, Lambda
    rows.sort(key=lambda r: (r["Re"], r["S"], r["Lambda"]))

    # Write CSV
    if rows:
        csv_path = outdir / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Re-entrant transition window expansion at N=256 (with resume)"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Max parallel workers (default: cpu count)"
    )
    parser.add_argument(
        "--cycles", type=int, default=20,
        help="Number of oscillation cycles (default: 20)"
    )
    parser.add_argument(
        "--outdir", type=str,
        default="results/reentrant-expansion",
        help="Output directory"
    )
    parser.add_argument(
        "--battery-pause", type=int, default=50,
        help="Pause when battery drops below this %% (default: 50, 0=disable)"
    )
    parser.add_argument(
        "--battery-resume", type=int, default=80,
        help="Resume when battery reaches this %% (default: 80)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print parameter count and exit without running cases"
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = configure_logging(args.outdir, "reentrant-expansion")

    # Build re-entrant expansion parameter grid
    all_params = build_reentrant_params()

    # Collect completed cases from all dedup sources + own output
    completed = set()
    dedup_sources = DEDUP_DIRS + [args.outdir]
    for dedup_dir in dedup_sources:
        dir_cases = load_completed_cases(dedup_dir)
        if dir_cases:
            print(f"  Found {len(dir_cases)} completed cases in {dedup_dir}")
        completed |= dir_cases

    n_completed_prior = len(completed)

    # Filter to only pending cases (using rounded keys for approximate float matching)
    pending_params = [
        p for p in all_params if _round_key(*p) not in completed
    ]

    n_total = len(all_params)
    n_pending = len(pending_params)
    n_deduped = n_total - n_pending

    print(f"\nRe-entrant expansion at N=256 (resumable)")
    print(f"Total cases in grid: {n_total}")
    print(f"Deduplicated (already exist): {n_deduped}")
    print(f"Remaining to run: {n_pending}")
    print(f"Grid: N=256, H=5.0, gamma=2.0, Sc_t=1.0, damping=linear")
    print(f"Cycles: {args.cycles}, Workers: {args.workers or 'auto'}")
    print()

    # Show per-(Re,S) breakdown
    for (Re, S), lambdas in sorted(REENTRANT_CASES.items()):
        n_in_group = len(lambdas)
        n_pending_in_group = sum(
            1 for lam in lambdas
            if _round_key(float(Re), float(S), float(lam)) not in completed
        )
        status = "all done" if n_pending_in_group == 0 else f"{n_pending_in_group} pending"
        print(f"  Re={Re:>4d}, S={S}: {n_in_group:>2d} cases ({status})")
    print()

    if args.dry_run:
        print("Dry run complete, exiting.")
        return

    if n_pending == 0:
        print("All cases already completed!")
        summary_rows = update_summary_csv(args.outdir)
        if summary_rows:
            n_laminar = sum(1 for r in summary_rows if r["regime"] == "laminar")
            n_turbulent = sum(1 for r in summary_rows if r["regime"] == "turbulent")
            print(f"Results: {n_laminar} laminar, {n_turbulent} turbulent")
        return

    # Estimate runtime
    est_time_per_case = 60  # seconds
    est_total_hours = n_pending * est_time_per_case / 3600
    print(f"Estimated remaining runtime: ~{est_total_hours:.1f} hours")
    print()

    # Build param dicts for pending cases
    param_dicts = [
        build_param_dict(Re, S, Lambda, N=256, n_cycles=args.cycles)
        for Re, S, Lambda in pending_params
    ]

    # Progress tracking
    try:
        from tqdm.auto import tqdm
        pbar = tqdm(total=n_pending, desc="Sweep", unit="case")
    except ImportError:
        pbar = None

    n_done = 0
    n_laminar = 0
    n_turbulent = 0

    # Check battery before starting (if enabled)
    if args.battery_pause > 0:
        pct, charging = get_battery_status()
        if pct is not None and pct < args.battery_pause:
            status = "charging" if charging else "discharging"
            print(f"Battery low ({pct}%, {status}), pausing before start...")
            wait_for_battery(min_percent=args.battery_resume)

    # Run with incremental saving - rolling window for battery-aware pause
    max_w = args.workers or os.cpu_count() or 1
    with ProcessPoolExecutor(max_workers=max_w) as pool:
        pending_iter = iter(param_dicts)
        active_futures = {}

        # Seed the pool with initial batch
        for p in itertools.islice(pending_iter, max_w):
            fut = pool.submit(single_run, p)
            active_futures[fut] = p

        while active_futures:
            # Wait for the next completed future
            for fut in as_completed(active_futures):
                break  # Get exactly one completed future

            params = active_futures.pop(fut)
            try:
                result = fut.result()

                # Save immediately
                p = result["params"]
                fname = make_filename(p["Re"], p["S"], p["Lambda"])
                save_run(result, outdir / fname)

                # Track statistics
                if result["regime"] == "laminar":
                    n_laminar += 1
                else:
                    n_turbulent += 1

                n_done += 1

                if pbar:
                    pbar.update(1)
                    pbar.set_postfix(L=n_laminar, T=n_turbulent)
                else:
                    print(f"  [{n_done}/{n_pending}] Re={p['Re']} S={p['S']} "
                          f"L={p['Lambda']:.4f} -> {result['regime']}")

            except Exception as e:
                p = params
                print(f"  ERROR: Re={p['Re']} S={p['S']} L={p['Lambda']} - {e}")

            # Check battery before submitting next job
            if args.battery_pause > 0:
                pct, charging = get_battery_status()
                if pct is not None and pct < args.battery_pause:
                    status = "charging" if charging else "discharging"
                    print(f"\n  Battery low ({pct}%, {status}), pausing...")
                    if pbar:
                        pbar.set_description("Paused (low battery)")
                    wait_for_battery(min_percent=args.battery_resume)
                    if pbar:
                        pbar.set_description("Sweep")

            # Submit next job if available
            next_p = next(pending_iter, None)
            if next_p is not None:
                fut = pool.submit(single_run, next_p)
                active_futures[fut] = next_p

    if pbar:
        pbar.close()

    # Update summary CSV with all results
    print("\nUpdating summary.csv...")
    summary_rows = update_summary_csv(args.outdir)

    # Final statistics
    total_laminar = sum(1 for r in summary_rows if r["regime"] == "laminar")
    total_turbulent = sum(1 for r in summary_rows if r["regime"] == "turbulent")
    total_converged = sum(1 for r in summary_rows if r["converged"])

    print()
    print(f"Session: {n_laminar} laminar, {n_turbulent} turbulent")
    print(f"Total: {total_laminar} laminar, {total_turbulent} turbulent")
    if summary_rows:
        print(f"Converged: {total_converged}/{len(summary_rows)} ({100*total_converged/len(summary_rows):.0f}%)")
    print(f"\nResults saved to {args.outdir}/")


if __name__ == "__main__":
    main()
