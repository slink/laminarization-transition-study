# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

"""Benchmarking utilities for profiling fluidflow hot paths.

Provides both micro-benchmarks (individual functions) and a macro-benchmark
(full single_run integration) with timing and optional cProfile output.
"""

import time
import cProfile
import pstats
import io
import numpy as np


def _make_test_data(N=128, H=5.0, gamma=2.0):
    """Create realistic test arrays for benchmarking."""
    from fluidflow.grid import StretchedGrid
    grid = StretchedGrid(N=N, H=H, gamma=gamma)
    u = np.exp(-grid.z) * np.sin(grid.z)
    u[0] = 0.0
    C = np.exp(-grid.z) * 0.5
    C[0] = 1.0
    return grid, u, C


def _time_fn(fn, args=(), kwargs=None, n_warmup=3, n_iter=100):
    """Time a function over n_iter calls, returning median and stats."""
    kwargs = kwargs or {}
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter_ns()
        fn(*args, **kwargs)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) * 1e-6)  # ms
    times = np.array(times)
    return {
        "median_ms": float(np.median(times)),
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "n_iter": n_iter,
    }


def bench_thomas_solve(N=128, n_iter=500):
    """Benchmark thomas_solve."""
    from fluidflow.solvers.tridiagonal import thomas_solve
    a = np.random.randn(N)
    b = np.random.randn(N) + 5.0  # diag dominant
    c = np.random.randn(N)
    d = np.random.randn(N)
    return _time_fn(thomas_solve, args=(a, b, c, d), n_iter=n_iter)


def bench_ddz(N=128, n_iter=500):
    """Benchmark ddz operator."""
    from fluidflow.solvers.operators import ddz
    grid, u, _ = _make_test_data(N)
    return _time_fn(ddz, args=(u, grid), n_iter=n_iter)


def bench_diffusion_matrix(N=128, n_iter=500):
    """Benchmark diffusion_operator_matrix."""
    from fluidflow.solvers.operators import diffusion_operator_matrix
    grid, u, _ = _make_test_data(N)
    nu = np.ones(N) * 0.01
    return _time_fn(diffusion_operator_matrix, args=(nu, grid), n_iter=n_iter)


def bench_diffusion_bands(N=128, n_iter=500):
    """Benchmark diffusion_operator_bands (tridiagonal band version)."""
    from fluidflow.solvers.operators import diffusion_operator_bands
    grid, u, _ = _make_test_data(N)
    nu = np.ones(N) * 0.01
    return _time_fn(diffusion_operator_bands, args=(nu, grid.z, grid.N), n_iter=n_iter)


def bench_compute_nu_t(N=128, n_iter=500):
    """Benchmark compute_nu_t."""
    from fluidflow.models.closures import compute_nu_t
    grid, u, C = _make_test_data(N)
    return _time_fn(compute_nu_t, args=(u, C, grid, 1.0), n_iter=n_iter)


def bench_model_step(N=128, n_iter=200):
    """Benchmark one OscillatoryBLModel.step() call."""
    from fluidflow.models.oscillatory_bl import OscillatoryBLModel
    params = dict(Re=500, S=0.01, Lambda=1.0, N=N, H=5.0, gamma=2.0)
    model = OscillatoryBLModel(params)
    u, C = model.get_initial_condition()
    t = 0.0

    def one_step():
        model.step(u, C, t)

    return _time_fn(one_step, n_iter=n_iter)


def bench_single_run(N=64, n_cycles=2):
    """Time a full single_run (macro benchmark)."""
    from fluidflow.sweep import single_run
    params = dict(Re=500, S=0.01, Lambda=1.0, N=N, H=5.0, gamma=2.0,
                  Sc_t=1.0, n_cycles=n_cycles, damping="linear")
    t0 = time.perf_counter()
    result = single_run(params)
    elapsed = time.perf_counter() - t0
    return {
        "elapsed_s": elapsed,
        "n_cycles_run": result["n_cycles_run"],
        "regime": result["regime"],
    }


def profile_single_run(N=64, n_cycles=2):
    """Run cProfile on single_run, return stats as string."""
    from fluidflow.sweep import single_run
    params = dict(Re=500, S=0.01, Lambda=1.0, N=N, H=5.0, gamma=2.0,
                  Sc_t=1.0, n_cycles=n_cycles, damping="linear")
    pr = cProfile.Profile()
    pr.enable()
    single_run(params)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    return s.getvalue()


def run_all_benchmarks(N=128, verbose=True):
    """Run all micro and macro benchmarks. Returns dict of results."""
    results = {}

    benches = [
        ("thomas_solve", bench_thomas_solve),
        ("ddz", bench_ddz),
        ("diffusion_matrix", bench_diffusion_matrix),
        ("compute_nu_t", bench_compute_nu_t),
        ("model_step", bench_model_step),
    ]

    # Check if bands version exists
    try:
        from fluidflow.solvers.operators import diffusion_operator_bands
        benches.insert(3, ("diffusion_bands", bench_diffusion_bands))
    except ImportError:
        pass

    for name, fn in benches:
        if verbose:
            print(f"  {name}...", end="", flush=True)
        r = fn(N=N)
        results[name] = r
        if verbose:
            print(f" {r['median_ms']:.3f} ms (median, n={r['n_iter']})")

    if verbose:
        print(f"  single_run (N=64, 2 cycles)...", end="", flush=True)
    r = bench_single_run(N=64, n_cycles=2)
    results["single_run"] = r
    if verbose:
        print(f" {r['elapsed_s']:.2f} s")

    return results


def compare_results(before, after):
    """Print a comparison table of two benchmark result sets."""
    print(f"\n{'Benchmark':<22} {'Before':>10} {'After':>10} {'Speedup':>10}")
    print("-" * 55)
    for key in before:
        if key == "single_run":
            b = before[key]["elapsed_s"]
            a = after[key]["elapsed_s"]
            speedup = b / a if a > 0 else float("inf")
            print(f"{key:<22} {b:>9.2f}s {a:>9.2f}s {speedup:>9.1f}x")
        else:
            b = before[key]["median_ms"]
            a = after[key]["median_ms"]
            speedup = b / a if a > 0 else float("inf")
            print(f"{key:<22} {b:>8.3f}ms {a:>8.3f}ms {speedup:>9.1f}x")


if __name__ == "__main__":
    print("=" * 55)
    print("FluidFlow Benchmarks")
    print("=" * 55)
    print()

    print("cProfile of single_run (N=64, 2 cycles):")
    print(profile_single_run())

    print("Micro-benchmarks (N=128):")
    run_all_benchmarks(N=128)
