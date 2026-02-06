#!/usr/bin/env python3
# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

"""Grid convergence verification at N=512 for near-transition cases.

Group 4 of the simulation plan: re-runs near-threshold cases (7 < viscosity_ratio < 15)
at N=512 with 40 cycles to verify that the transition boundary identified at N=256 is
grid-converged. Also always includes the 3 re-entrant profile cases regardless of their
viscosity ratio.

Sources for N=256 results:
- results/phase1-n256/           (Phase 1 reconnaissance, 270 cases)
- results/archive/phase3-n256/   (Phase 3 archived data)
- results/reentrant-expansion/   (Group 3 expansion data)

Features:
- Incremental saving: results saved as each case completes
- Resume capability: skips already-completed cases on restart
- Parallel execution with progress tracking
- Battery-aware pause/resume (macOS only): pauses when battery low, resumes when charged

Usage:
    python scripts/run_n512_production.py [--workers N] [--battery-pause PCT]

Examples:
    # Preview which cases will run
    python scripts/run_n512_production.py --dry-run

    # Run all near-transition cases at N=512
    python scripts/run_n512_production.py

    # Run with 4 workers and 50 cycles
    python scripts/run_n512_production.py --workers 4 --cycles 50

    # Resume after interruption (just run the same command)
    python scripts/run_n512_production.py

    # Disable battery monitoring
    python scripts/run_n512_production.py --battery-pause 0
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

# --- Source directories for N=256 results ---
N256_SOURCE_DIRS = [
    "results/phase1-n256",
    "results/archive/phase3-n256",
    "results/reentrant-expansion",
]

# --- Re-entrant profile cases to always include ---
REENTRANT_PROFILES = [
    (1000.0, 0.005, 0.1),
    (1000.0, 0.005, 2.0),
    (1000.0, 0.005, 5.0),
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


def load_n256_near_transition(dirs, threshold_low=7, threshold_high=15):
    """Load N=256 results and filter to near-transition cases.

    Scans all JSON result files in the given directories and returns parameter
    tuples where the viscosity_ratio falls within the near-threshold window
    (threshold_low < viscosity_ratio < threshold_high).

    Args:
        dirs: list of directory paths containing N=256 JSON results
        threshold_low: lower bound on viscosity_ratio (exclusive)
        threshold_high: upper bound on viscosity_ratio (exclusive)

    Returns:
        (near_transition_params, per_dir_counts) where:
        - near_transition_params: list of (Re, S, Lambda) tuples
        - per_dir_counts: dict mapping dir -> (total_loaded, n_near_transition)
    """
    seen = set()  # deduplicate across source directories
    near_transition = []
    per_dir_counts = {}

    for d in dirs:
        dirpath = Path(d)
        if not dirpath.exists():
            per_dir_counts[d] = (0, 0)
            continue

        n_loaded = 0
        n_near = 0

        for json_file in dirpath.glob("Re*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                p = data["params"]
                vr = data.get("viscosity_ratio")
                if vr is None:
                    continue

                n_loaded += 1
                key = _round_key(p["Re"], p["S"], p["Lambda"])

                if key in seen:
                    continue
                seen.add(key)

                if threshold_low < vr < threshold_high:
                    near_transition.append((float(p["Re"]), float(p["S"]), float(p["Lambda"])))
                    n_near += 1

            except (json.JSONDecodeError, KeyError):
                continue

        per_dir_counts[d] = (n_loaded, n_near)

    return near_transition, per_dir_counts


def build_param_dict(Re, S, Lambda, N=512, H=5.0, gamma=2.0, Sc_t=1.0, n_cycles=40):
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
        description="N=512 grid convergence verification for near-transition cases (with resume)"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Max parallel workers (default: cpu count)"
    )
    parser.add_argument(
        "--cycles", type=int, default=40,
        help="Number of oscillation cycles (default: 40)"
    )
    parser.add_argument(
        "--outdir", type=str,
        default="results/n512-production",
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
    logger = configure_logging(args.outdir, "n512-production")

    # --- Step 1: Load near-transition cases from N=256 results ---
    print("Loading N=256 results and filtering near-transition cases...")
    print(f"  Threshold: 7 < viscosity_ratio < 15\n")

    near_transition, per_dir_counts = load_n256_near_transition(N256_SOURCE_DIRS)

    total_loaded = 0
    for d in N256_SOURCE_DIRS:
        n_loaded, n_near = per_dir_counts.get(d, (0, 0))
        total_loaded += n_loaded
        status = f"{n_loaded} loaded, {n_near} near-transition"
        if n_loaded == 0:
            status = "directory not found or empty"
        print(f"  {d}: {status}")

    print(f"\n  Total loaded from sources: {total_loaded}")
    print(f"  Near-transition cases (7 < vr < 15): {len(near_transition)}")

    # --- Step 2: Add forced re-entrant profile cases ---
    near_transition_keys = {_round_key(*p) for p in near_transition}
    n_reentrant_added = 0
    for rp in REENTRANT_PROFILES:
        key = _round_key(*rp)
        if key not in near_transition_keys:
            near_transition.append(rp)
            near_transition_keys.add(key)
            n_reentrant_added += 1

    print(f"  Forced re-entrant profile inclusions: {n_reentrant_added} new "
          f"(of {len(REENTRANT_PROFILES)} total, rest already in near-transition set)")

    # Deduplicate the combined list (should already be unique via key checking)
    all_params = near_transition

    # --- Step 3: Check for already-completed N=512 cases ---
    completed = load_completed_cases(args.outdir)
    n_already_done = len(completed)

    # Filter to only pending cases
    pending_params = [
        p for p in all_params if _round_key(*p) not in completed
    ]

    n_total = len(all_params)
    n_pending = len(pending_params)

    print(f"\n  Already completed at N=512: {n_already_done}")
    print(f"  Total to run: {n_total}")
    print(f"  Remaining after dedup: {n_pending}")

    print(f"\nN=512 grid convergence (resumable)")
    print(f"Grid: N=512, H=5.0, gamma=2.0, Sc_t=1.0, damping=linear")
    print(f"Cycles: {args.cycles}, Workers: {args.workers or 'auto'}")
    print()

    if args.dry_run:
        # Show all cases that would run
        print("Cases to run:")
        for Re, S, Lambda in sorted(pending_params):
            reentrant_tag = " [re-entrant]" if _round_key(Re, S, Lambda) in {_round_key(*rp) for rp in REENTRANT_PROFILES} else ""
            print(f"  Re={Re:>6.0f}  S={S:<8.4f}  Lambda={Lambda:<8.4f}{reentrant_tag}")
        print(f"\nDry run complete, {n_pending} cases would be submitted.")
        return

    if n_pending == 0:
        print("All cases already completed!")
        summary_rows = update_summary_csv(args.outdir)
        if summary_rows:
            n_laminar = sum(1 for r in summary_rows if r["regime"] == "laminar")
            n_turbulent = sum(1 for r in summary_rows if r["regime"] == "turbulent")
            print(f"Results: {n_laminar} laminar, {n_turbulent} turbulent")
        return

    # Estimate runtime (N=512 cases are ~4x more expensive than N=256)
    est_time_per_case = 240  # seconds
    est_total_hours = n_pending * est_time_per_case / 3600
    print(f"Estimated remaining runtime: ~{est_total_hours:.1f} hours")
    print()

    # Build param dicts for pending cases
    param_dicts = [
        build_param_dict(Re, S, Lambda, N=512, n_cycles=args.cycles)
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
