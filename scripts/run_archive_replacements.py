#!/usr/bin/env python3
# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

"""Re-run archive cases into active data directories.

Replaces all archive data dependencies with fresh runs into active directories:

Group A - Production diagnostics (30 cases):
    Re-run the exact 30 (Re, S, Lambda) combos from archive/phase4-production-revised
    with N=256, 40 cycles, linear damping.
    Output: results/production-n256/

Group B - Grid convergence (24 cases):
    Re-run 6 parameter combos x 4 grid sizes (N=32, 64, 128, 256).
    Output: results/grid-convergence/N{32,64,128,256}/

Group C - Re-entrant vertical profiles (3 cases):
    Re-run (Re=1000, S=0.005, Lambda=0.1/2.0/5.0) with N=256, 20 cycles.
    Output: results/reentrant-profiles/

Features:
- Incremental saving with resume capability
- Parallel execution with progress tracking
- Battery-aware pause/resume (macOS only)

Usage:
    python scripts/run_archive_replacements.py [--workers N] [--group A|B|C|all]
    python scripts/run_archive_replacements.py --dry-run
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

# --- Group A: Production diagnostics (30 cases) ---
# Exact parameters from archive/phase4-production-revised
PRODUCTION_CASES = [
    (300,  0.005, 0.25),
    (300,  0.005, 0.75),
    (300,  0.005, 1.25),
    (300,  0.005, 2.5),
    (500,  0.005, 0.1),
    (500,  0.005, 0.5),
    (500,  0.005, 1.25),
    (500,  0.005, 2.0),
    (500,  0.005, 3.0),
    (500,  0.01,  0.25),
    (500,  0.01,  0.75),
    (500,  0.01,  1.25),
    (500,  0.01,  2.5),
    (750,  0.005, 0.1),
    (750,  0.005, 0.5),
    (750,  0.005, 1.5),
    (750,  0.005, 3.0),
    (750,  0.01,  0.1),
    (750,  0.01,  0.5),
    (750,  0.01,  1.5),
    (750,  0.01,  3.5),
    (1000, 0.005, 0.1),
    (1000, 0.005, 0.5),
    (1000, 0.005, 1.5),
    (1000, 0.005, 3.0),
    (1000, 0.005, 5.0),
    (1000, 0.01,  0.1),
    (1000, 0.01,  0.5),
    (1000, 0.01,  1.5),
    (1000, 0.01,  3.0),
]

# --- Group B: Grid convergence (6 combos x 4 grid sizes = 24 cases) ---
CONVERGENCE_PARAMS = [
    (100,  0.1,  0.1),
    (300,  0.05, 0.5),
    (500,  0.01, 0.1),
    (500,  0.1,  0.0),
    (500,  0.1,  0.5),
    (1000, 0.1,  5.0),
]
CONVERGENCE_GRIDS = [32, 64, 128, 256]

# --- Group C: Re-entrant vertical profiles (3 cases) ---
REENTRANT_PROFILE_CASES = [
    (1000, 0.005, 0.1),
    (1000, 0.005, 2.0),
    (1000, 0.005, 5.0),
]


def get_battery_status():
    """Get battery percentage and charging status on macOS."""
    try:
        result = subprocess.run(
            ["pmset", "-g", "batt"],
            capture_output=True, text=True, timeout=5
        )
        output = result.stdout
        match = re.search(r'(\d+)%', output)
        if not match:
            return (None, None)
        percentage = int(match.group(1))
        on_ac = "AC Power" in output
        is_charging = on_ac or (
            "charging" in output.lower() and "discharging" not in output.lower()
        )
        return (percentage, is_charging)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return (None, None)


def wait_for_battery(min_percent=80, check_interval=60):
    """Wait until battery percentage is above threshold."""
    while True:
        pct, charging = get_battery_status()
        if pct is None:
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
            continue
    return completed


def build_param_dict(Re, S, Lambda, N=256, H=5.0, gamma=2.0, Sc_t=1.0,
                     n_cycles=20, damping="linear"):
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
        "damping": damping,
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
    rows.sort(key=lambda r: (r["Re"], r["S"], r["Lambda"]))
    if rows:
        csv_path = outdir / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return rows


def run_cases(param_dicts, outdir, label, args):
    """Run a set of cases with incremental saving and battery awareness."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Check what's already done in this output dir
    completed = load_completed_cases(outdir)

    # Filter pending
    pending = []
    for pd in param_dicts:
        key = _round_key(pd["Re"], pd["S"], pd["Lambda"])
        if key not in completed:
            pending.append(pd)

    n_total = len(param_dicts)
    n_pending = len(pending)
    n_done_prior = n_total - n_pending

    print(f"\n--- {label} ---")
    print(f"  Total: {n_total}, Already done: {n_done_prior}, Pending: {n_pending}")

    if args.dry_run:
        for pd in param_dicts:
            status = "done" if _round_key(pd["Re"], pd["S"], pd["Lambda"]) in completed else "pending"
            print(f"    Re={pd['Re']}, S={pd['S']}, Lambda={pd['Lambda']}, N={pd['N']} [{status}]")
        return

    if n_pending == 0:
        print("  All cases already completed!")
        return

    # Progress tracking
    try:
        from tqdm.auto import tqdm
        pbar = tqdm(total=n_pending, desc=label, unit="case")
    except ImportError:
        pbar = None

    n_done = 0
    n_laminar = 0
    n_turbulent = 0

    # Check battery before starting
    if args.battery_pause > 0:
        pct, charging = get_battery_status()
        if pct is not None and pct < args.battery_pause:
            print(f"  Battery low ({pct}%), pausing...")
            wait_for_battery(min_percent=args.battery_resume)

    max_w = args.workers or os.cpu_count() or 1
    with ProcessPoolExecutor(max_workers=max_w) as pool:
        pending_iter = iter(pending)
        active_futures = {}

        for p in itertools.islice(pending_iter, max_w):
            fut = pool.submit(single_run, p)
            active_futures[fut] = p

        while active_futures:
            for fut in as_completed(active_futures):
                break

            params = active_futures.pop(fut)
            try:
                result = fut.result()
                p = result["params"]
                fname = make_filename(p["Re"], p["S"], p["Lambda"])
                save_run(result, outdir / fname)

                if result["regime"] == "laminar":
                    n_laminar += 1
                else:
                    n_turbulent += 1
                n_done += 1

                if pbar:
                    pbar.update(1)
                    pbar.set_postfix(L=n_laminar, T=n_turbulent)
                else:
                    n_str = f"N={params['N']}" if params.get('N', 256) != 256 else ""
                    print(f"  [{n_done}/{n_pending}] Re={p['Re']} S={p['S']} "
                          f"L={p['Lambda']:.4f} {n_str} -> {result['regime']}")

            except Exception as e:
                print(f"  ERROR: Re={params['Re']} S={params['S']} "
                      f"L={params['Lambda']} - {e}")

            # Battery check
            if args.battery_pause > 0:
                pct, charging = get_battery_status()
                if pct is not None and pct < args.battery_pause:
                    print(f"\n  Battery low ({pct}%), pausing...")
                    if pbar:
                        pbar.set_description("Paused (low battery)")
                    wait_for_battery(min_percent=args.battery_resume)
                    if pbar:
                        pbar.set_description(label)

            next_p = next(pending_iter, None)
            if next_p is not None:
                fut = pool.submit(single_run, next_p)
                active_futures[fut] = next_p

    if pbar:
        pbar.close()

    print(f"  Session: {n_laminar} laminar, {n_turbulent} turbulent")


def main():
    parser = argparse.ArgumentParser(
        description="Re-run archive cases into active data directories (with resume)"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Max parallel workers (default: cpu count)"
    )
    parser.add_argument(
        "--group", type=str, default="all",
        choices=["A", "B", "C", "all"],
        help="Which group to run: A=production, B=convergence, C=profiles, all=everything"
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

    print("Archive replacement re-runs")
    print("===========================")

    groups_to_run = ["A", "B", "C"] if args.group == "all" else [args.group]

    # --- Group A: Production diagnostics ---
    if "A" in groups_to_run:
        param_dicts = [
            build_param_dict(Re, S, Lambda, N=256, n_cycles=40)
            for Re, S, Lambda in PRODUCTION_CASES
        ]
        outdir = "results/production-n256"
        logger = configure_logging(outdir, "production-n256")
        run_cases(param_dicts, outdir, "Group A: Production diagnostics", args)
        if not args.dry_run:
            update_summary_csv(outdir)

    # --- Group B: Grid convergence ---
    if "B" in groups_to_run:
        for N in CONVERGENCE_GRIDS:
            param_dicts = [
                build_param_dict(Re, S, Lambda, N=N, n_cycles=20)
                for Re, S, Lambda in CONVERGENCE_PARAMS
            ]
            outdir = f"results/grid-convergence/N{N}"
            logger = configure_logging(outdir, f"grid-convergence-N{N}")
            run_cases(param_dicts, outdir, f"Group B: Grid convergence N={N}", args)
            if not args.dry_run:
                update_summary_csv(outdir)

    # --- Group C: Re-entrant profiles ---
    if "C" in groups_to_run:
        param_dicts = [
            build_param_dict(Re, S, Lambda, N=256, n_cycles=20)
            for Re, S, Lambda in REENTRANT_PROFILE_CASES
        ]
        outdir = "results/reentrant-profiles"
        logger = configure_logging(outdir, "reentrant-profiles")
        run_cases(param_dicts, outdir, "Group C: Re-entrant profiles", args)
        if not args.dry_run:
            update_summary_csv(outdir)

    if args.dry_run:
        print("\nDry run complete, exiting.")
    else:
        print("\nAll groups complete.")


if __name__ == "__main__":
    main()
