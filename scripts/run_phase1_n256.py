#!/usr/bin/env python3
# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

"""Re-run Phase 1 reconnaissance sweep at N=256.

Phase 1 was originally run at N=64 (270 cases across Re x S x Lambda space).
This script re-runs the same parameter grid at higher resolution (N=256) to
produce publication-quality results for the phase map and scaling laws.

Features:
- Incremental saving: results saved as each case completes
- Resume capability: skips already-completed cases on restart
- Parallel execution with progress tracking
- Battery-aware pause/resume (macOS only): pauses when battery low, resumes when charged

Usage:
    python scripts/run_phase1_n256.py [--workers N] [--battery-pause PCT]

Examples:
    # Run all 270 Phase 1 cases at N=256
    python scripts/run_phase1_n256.py

    # Run with 4 workers
    python scripts/run_phase1_n256.py --workers 4

    # Resume after interruption (just run the same command)
    python scripts/run_phase1_n256.py

    # Disable battery monitoring
    python scripts/run_phase1_n256.py --battery-pause 0
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
from itertools import product
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fluidflow.sweep import single_run
from fluidflow.io import save_run
from fluidflow.sweep_utils import configure_logging

# --- Parameters from sweep-plan.md Phase 1 (270 cases) ---
RE_VALS = [100, 200, 300, 500, 750, 1000]
S_VALS = [0.005, 0.01, 0.05, 0.1, 0.5]
LAMBDA_VALS = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]


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


def load_completed_cases(outdir):
    """Load set of (Re, S, Lambda) tuples that are already completed."""
    completed = set()
    outdir = Path(outdir)
    if not outdir.exists():
        return completed

    for json_file in outdir.glob("Re*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            p = data["params"]
            completed.add((p["Re"], p["S"], p["Lambda"]))
        except (json.JSONDecodeError, KeyError):
            # Skip malformed files
            continue

    return completed


def build_phase1_params():
    """Build Phase 1 parameter grid as list of (Re, S, Lambda) tuples."""
    return list(product(RE_VALS, S_VALS, LAMBDA_VALS))


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
        description="Re-run Phase 1 reconnaissance sweep at N=256 (with resume)"
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
        default="results/phase1-n256",
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
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = configure_logging(args.outdir, "phase1-n256")

    # Build Phase 1 parameter grid (Cartesian product)
    all_params = build_phase1_params()

    # Check for already-completed cases
    completed = load_completed_cases(args.outdir)
    n_completed_prior = len(completed)

    # Filter to only pending cases
    pending_params = [p for p in all_params if p not in completed]

    n_total = len(all_params)
    n_pending = len(pending_params)

    print(f"Phase 1 at N=256 (resumable)")
    print(f"Total cases: {n_total}")
    print(f"Already completed: {n_completed_prior}")
    print(f"Remaining: {n_pending}")
    print(f"Grid: N=256, H=5.0, gamma=2.0, Sc_t=1.0")
    print(f"Cycles: {args.cycles}, Workers: {args.workers or 'auto'}")
    print()

    if n_pending == 0:
        print("All cases already completed!")
        summary_rows = update_summary_csv(args.outdir)
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
                          f"L={p['Lambda']:.3f} -> {result['regime']}")

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
    print(f"Converged: {total_converged}/{len(summary_rows)} ({100*total_converged/len(summary_rows):.0f}%)")
    print(f"\nResults saved to {args.outdir}/")


if __name__ == "__main__":
    main()
