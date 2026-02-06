# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

# src/fluidflow/sweep.py
import logging
import numpy as np
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from fluidflow.models.oscillatory_bl import OscillatoryBLModel
from fluidflow.models.closures import compute_nu_t
from fluidflow.diagnostics import CycleDiagnostics

logger = logging.getLogger(__name__)


def single_run(params):
    """Run a single (Re, S, Lambda) case to periodic steady state.

    Args:
        params: dict with Re, S, Lambda, N, H, gamma, Sc_t, n_cycles.

    Returns:
        dict with params, diagnostics, regime label, convergence info.
    """
    n_cycles = params.get("n_cycles", 20)
    tol = params.get("tol", 0.01)

    model = OscillatoryBLModel(params)
    u, C = model.get_initial_condition()

    T_cycle = 2 * np.pi / model.omega
    steps_per_cycle = int(np.ceil(T_cycle / model.dt))

    t = 0.0
    prev_visc_ratio = None
    converged = False

    for cycle in range(n_cycles):
        # Diagnostics for this cycle
        diag = CycleDiagnostics(model.grid, g_prime=model.g_prime)
        sample_interval = max(1, steps_per_cycle // 20)  # ~20 samples per cycle

        for step_i in range(steps_per_cycle):
            nu_t, D_t = compute_nu_t(
                u, C, model.grid,
                g_prime=model.g_prime,
                Sc_t=model.Sc_t,
                damping=model.damping,
            )
            if step_i % sample_interval == 0:
                diag.accumulate(u, C, nu_t, D_t, model.nu, t)
            u, C = model.step(u, C, t)
            t += model.dt

        result = diag.finalize()
        visc_ratio = result["viscosity_ratio"]

        # Check convergence
        if prev_visc_ratio is not None and prev_visc_ratio > 0:
            change = abs(visc_ratio - prev_visc_ratio) / max(prev_visc_ratio, 1e-10)
            if change < tol:
                converged = True
                break
        prev_visc_ratio = visc_ratio

    result["params"] = {k: params[k] for k in ["Re", "S", "Lambda"]}
    result["converged"] = converged
    result["n_cycles_run"] = cycle + 1

    return result


def build_sweep_grid(Re_vals, S_vals, Lambda_vals, N=128, H=5.0, gamma=2.0,
                     Sc_t=1.0, n_cycles=20, damping="linear"):
    """Build list of parameter dicts for a full sweep."""
    grid = []
    for Re, S, Lambda in product(Re_vals, S_vals, Lambda_vals):
        grid.append(dict(
            Re=Re, S=S, Lambda=Lambda,
            N=N, H=H, gamma=gamma, Sc_t=Sc_t, n_cycles=n_cycles,
            damping=damping,
        ))
    return grid


def run_sweep(param_list, max_workers=None, progress=True):
    """Run parameter sweep in parallel.

    Args:
        param_list: list of param dicts from build_sweep_grid.
        max_workers: number of parallel processes (None = cpu count).
        progress: show tqdm progress bar if available.

    Returns:
        list of result dicts, in the same order as param_list.
    """
    n = len(param_list)
    logger.info("Starting sweep: %d cases, max_workers=%s", n, max_workers)

    # Soft import of tqdm
    tqdm_bar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            tqdm_bar = tqdm(total=n, desc="Sweep", unit="case")
        except ImportError:
            pass

    results = [None] * n

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_to_index = {}
        for i, params in enumerate(param_list):
            future = pool.submit(single_run, params)
            future_to_index[future] = i

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            results[idx] = future.result()
            p = results[idx]["params"]
            logger.debug(
                "Case %d/%d done: Re=%s S=%s Lambda=%s -> %s",
                idx + 1, n, p["Re"], p["S"], p["Lambda"],
                results[idx]["regime"],
            )
            if tqdm_bar is not None:
                tqdm_bar.update(1)

    if tqdm_bar is not None:
        tqdm_bar.close()

    logger.info("Sweep complete: %d cases finished", n)
    return results
