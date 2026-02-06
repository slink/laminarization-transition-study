#!/usr/bin/env python
"""Reprocess all result JSONs with corrected viscosity ratio.

The viscosity ratio was previously computed as np.mean(nu_t_avg) / nu,
which gives unequal physical weighting on a stretched grid. This script
recomputes it using trapezoidal integration:

    <nu_t> = (1/H) * integral_0^H nu_t(z) dz

and updates the regime classification accordingly.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

THRESHOLD = 10.0

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

PHASE_DIRS = [
    "phase1-coarse",
    "phase2-convergence/N32",
    "phase2-convergence/N64",
    "phase2-convergence/N128",
    "phase2-convergence/N256",
    "phase3-refined",
    "phase4-production",
    "reentrant_profiles",
    "exponential",
]


def reprocess_json(filepath):
    """Recompute viscosity_ratio and regime from stored profiles.

    Returns None if the file lacks stored profiles (can't recompute).
    """
    with open(filepath) as f:
        data = json.load(f)

    if "profiles" not in data:
        return None

    nu_t = np.array(data["profiles"]["nu_t"])
    z = np.array(data["profiles"]["z"])
    Re = data["params"]["Re"]
    nu = 1.0 / Re

    old_vr = data["viscosity_ratio"]
    old_regime = data["regime"]

    new_vr = float(np.trapezoid(nu_t, z) / z[-1] / nu)
    new_regime = "turbulent" if new_vr > THRESHOLD else "laminar"

    data["viscosity_ratio"] = new_vr
    data["regime"] = new_regime

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return old_vr, new_vr, old_regime, new_regime


def regenerate_summary_csv(phase_dir):
    """Regenerate summary.csv from all JSONs in a phase directory."""
    json_files = sorted(phase_dir.glob("*.json"))
    if not json_files:
        return

    rows = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        rows.append({
            "Re": data["params"]["Re"],
            "S": data["params"]["S"],
            "Lambda": data["params"]["Lambda"],
            "regime": data["regime"],
            "converged": str(data["converged"]),
            "viscosity_ratio": data["viscosity_ratio"],
        })

    csv_path = phase_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Re", "S", "Lambda", "regime",
                                                "converged", "viscosity_ratio"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    total_files = 0
    regime_changes = 0

    for phase_name in PHASE_DIRS:
        phase_dir = RESULTS_DIR / phase_name
        if not phase_dir.exists():
            print(f"  SKIP {phase_name} (not found)")
            continue

        json_files = sorted(phase_dir.glob("*.json"))
        changed = 0
        skipped = 0

        for jf in json_files:
            result = reprocess_json(jf)
            if result is None:
                skipped += 1
                continue

            old_vr, new_vr, old_regime, new_regime = result
            total_files += 1

            if old_regime != new_regime:
                regime_changes += 1
                changed += 1
                print(f"  REGIME CHANGE: {jf.name}: {old_regime} -> {new_regime} "
                      f"(vr: {old_vr:.2f} -> {new_vr:.2f})")

        regenerate_summary_csv(phase_dir)
        msg = f"  {phase_name}: {len(json_files) - skipped} reprocessed, {changed} regime changes"
        if skipped:
            msg += f", {skipped} skipped (no profiles)"
        print(msg)

    print(f"\nTotal: {total_files} files reprocessed, {regime_changes} regime changes")


if __name__ == "__main__":
    main()
