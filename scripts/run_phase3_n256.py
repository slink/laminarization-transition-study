#!/usr/bin/env python3
# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

"""Re-run Phase 3 refined transition search at N=256.

Phase 3 was originally run at N=128. This script re-runs the same parameter
points at higher resolution to verify transition boundaries, particularly
for S=0.01 cases where Phase 4 results differed from Phase 3 predictions.

Features:
- Incremental saving: results saved as each case completes
- Resume capability: skips already-completed cases on restart
- Parallel execution with progress tracking
- Battery-aware pause/resume (macOS only): pauses when battery low, resumes when charged

Usage:
    python scripts/run_phase3_n256.py [--workers N] [--subset S_VALUES]

Examples:
    # Run all Phase 3 cases at N=256
    python scripts/run_phase3_n256.py

    # Run only S=0.005 and S=0.01 cases (used in Table II)
    python scripts/run_phase3_n256.py --subset 0.005,0.01

    # Resume after interruption (just run the same command)
    python scripts/run_phase3_n256.py --subset 0.005,0.01

    # Disable battery monitoring
    python scripts/run_phase3_n256.py --battery-pause 0
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

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fluidflow.sweep import single_run
from fluidflow.io import save_run
from fluidflow.sweep_utils import configure_logging


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


def load_phase3_params(csv_path, subset_S=None):
    """Load Phase 3 parameter points from summary CSV."""
    params = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            Re = float(row["Re"])
            S = float(row["S"])
            Lambda = float(row["Lambda"])

            if subset_S is None or S in subset_S:
                params.append((Re, S, Lambda))

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
        description="Re-run Phase 3 refined transition search at N=256 (with resume)"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Max parallel workers (default: cpu count)"
    )
    parser.add_argument(
        "--subset", type=str, default=None,
        help="Comma-separated S values to run (e.g., '0.005,0.01'). Default: all"
    )
    parser.add_argument(
        "--cycles", type=int, default=20,
        help="Number of oscillation cycles (default: 20)"
    )
    parser.add_argument(
        "--input", type=str,
        default="results/phase3-refined/summary.csv",
        help="Path to original Phase 3 summary CSV"
    )
    parser.add_argument(
        "--outdir", type=str,
        default="results/phase3-n256",
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

    # Parse subset S values
    subset_S = None
    if args.subset:
        subset_S = [float(s.strip()) for s in args.subset.split(",")]

    # Setup logging
    logger = configure_logging(args.outdir, "phase3-n256")

    # Load all Phase 3 parameters
    all_params = load_phase3_params(args.input, subset_S)

    # Check for already-completed cases
    completed = load_completed_cases(args.outdir)
    n_completed_prior = len(completed)

    # Filter to only pending cases
    pending_params = [p for p in all_params if p not in completed]

    n_total = len(all_params)
    n_pending = len(pending_params)

    print(f"Phase 3 at N=256 (resumable)")
    print(f"Total cases: {n_total}")
    print(f"Already completed: {n_completed_prior}")
    print(f"Remaining: {n_pending}")
    if subset_S:
        print(f"Filtering to S values: {subset_S}")
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
