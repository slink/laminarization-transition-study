# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

"""Shared utilities for sweep scripts: save/load results, logging, summary tables."""

import csv
import logging
import os

from fluidflow.io import save_run


def save_sweep_results(results, outdir):
    """Save per-case JSON files and a summary CSV.

    Args:
        results: list of result dicts from single_run.
        outdir: output directory path.

    Returns:
        list of summary row dicts.
    """
    os.makedirs(outdir, exist_ok=True)
    summary_rows = []

    for r in results:
        p = r["params"]
        fname = f"Re{p['Re']}_S{p['S']}_L{p['Lambda']}.json"
        save_run(r, os.path.join(outdir, fname))

        summary_rows.append({
            "Re": p["Re"],
            "S": p["S"],
            "Lambda": p["Lambda"],
            "regime": r["regime"],
            "converged": r["converged"],
            "viscosity_ratio": r["viscosity_ratio"],
        })

    # Write summary CSV
    if summary_rows:
        csv_path = os.path.join(outdir, "summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

    return summary_rows


def load_phase_summary(csv_path):
    """Read a summary CSV into a list of dicts with proper types.

    Numeric columns (Re, S, Lambda, viscosity_ratio) are converted to float.
    The 'converged' column is converted to bool.
    """
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            typed = {}
            for k, v in row.items():
                if k in ("Re", "S", "Lambda", "viscosity_ratio"):
                    typed[k] = float(v)
                elif k == "converged":
                    typed[k] = v.strip().lower() in ("true", "1", "yes")
                else:
                    typed[k] = v
            rows.append(typed)
    return rows


def configure_logging(outdir, phase_name):
    """Set up file + console logging on the 'fluidflow' logger.

    Args:
        outdir: directory for the log file.
        phase_name: used in the log filename.

    Returns:
        the configured logger.
    """
    os.makedirs(outdir, exist_ok=True)
    logger = logging.getLogger("fluidflow")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    # File handler
    log_path = os.path.join(outdir, f"{phase_name}.log")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler (only if none already exists)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


def print_summary_table(summary_rows):
    """Print a formatted summary table to stdout."""
    header = f"{'Re':>8} {'S':>8} {'Lambda':>8} {'regime':>10} {'conv':>6} {'nu_t/nu':>10}"
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['Re']:>8.1f} {row['S']:>8.4f} {row['Lambda']:>8.4f} "
            f"{row['regime']:>10} {str(row['converged']):>6} "
            f"{row['viscosity_ratio']:>10.4f}"
        )
