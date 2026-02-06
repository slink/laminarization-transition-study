#!/usr/bin/env python
"""Validate Numba JIT regression: re-run 6 Phase 2 reference cases and compare.

Loads archived Phase 2 N=256 results, re-runs each case with the current solver,
and asserts viscosity_ratio agreement within 10% and exact regime match.

The 10% tolerance accounts for floating-point reordering: the Numba PR replaced
diffusion_operator_matrix (dense NxN) with diffusion_operator_bands (tridiagonal),
changing FP accumulation order across thousands of timesteps. Regime classification
(the scientifically meaningful output) must match exactly.
"""

import glob
import json
import os
import sys
import time

# Allow running without pip install -e .
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fluidflow.sweep import single_run

ARCHIVE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "archive", "phase2-convergence", "N256"
)
REL_TOL = 0.10  # 10% relative tolerance (FP reordering from Numba bands)


def main():
    json_files = sorted(glob.glob(os.path.join(ARCHIVE_DIR, "Re*.json")))
    if not json_files:
        print(f"ERROR: No JSON files found in {ARCHIVE_DIR}")
        sys.exit(1)

    print(f"Numba regression validation: {len(json_files)} cases from Phase 2 N=256")
    print("=" * 80)

    all_pass = True
    rows = []

    for path in json_files:
        with open(path) as f:
            ref = json.load(f)

        p = ref["params"]
        label = f"Re={p['Re']}, S={p['S']}, Lambda={p['Lambda']}"
        print(f"\nRunning {label} ...")

        params = dict(
            Re=p["Re"], S=p["S"], Lambda=p["Lambda"],
            N=256, H=5.0, gamma=2.0, Sc_t=1.0, n_cycles=20, damping="linear",
        )

        t0 = time.time()
        result = single_run(params)
        elapsed = time.time() - t0

        old_vr = ref["viscosity_ratio"]
        new_vr = result["viscosity_ratio"]
        rel_err = abs(new_vr - old_vr) / max(abs(old_vr), 1e-12)
        vr_ok = rel_err < REL_TOL

        old_regime = ref["regime"]
        new_regime = result["regime"]
        regime_ok = old_regime == new_regime

        case_pass = vr_ok and regime_ok
        if not case_pass:
            all_pass = False

        rows.append((label, old_vr, new_vr, rel_err, old_regime, new_regime, case_pass, elapsed))
        status = "PASS" if case_pass else "FAIL"
        print(f"  {status}  vr: {old_vr:.6f} -> {new_vr:.6f}  (err={rel_err:.2e})  "
              f"regime: {old_regime} -> {new_regime}  [{elapsed:.1f}s]")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Case':<35} {'Old VR':>10} {'New VR':>10} {'Rel Err':>10} "
          f"{'Regime':>12} {'Status':>8}")
    print("-" * 80)
    for label, old_vr, new_vr, rel_err, old_reg, new_reg, ok, elapsed in rows:
        regime_str = old_reg if old_reg == new_reg else f"{old_reg}->{new_reg}"
        print(f"{label:<35} {old_vr:>10.4f} {new_vr:>10.4f} {rel_err:>10.2e} "
              f"{regime_str:>12} {'PASS' if ok else 'FAIL':>8}")
    print("=" * 80)

    n_pass = sum(1 for r in rows if r[6])
    print(f"\nResult: {n_pass}/{len(rows)} cases passed")

    if all_pass:
        print("All regression checks PASSED.")
    else:
        print("REGRESSION FAILURE: some cases did not match archived results.")
        sys.exit(1)


if __name__ == "__main__":
    main()
