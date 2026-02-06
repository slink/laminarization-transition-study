# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

import numpy as np
from fluidflow.solvers.tridiagonal import thomas_solve

def test_identity_system():
    """Solving I*x = b should return b."""
    N = 10
    a = np.zeros(N)
    b = np.ones(N)
    c = np.zeros(N)
    d = np.arange(N, dtype=float)
    x = thomas_solve(a, b, c, d)
    assert np.allclose(x, d)

def test_tridiagonal_vs_numpy():
    """Thomas solve should match np.linalg.solve for a random tridiagonal."""
    rng = np.random.default_rng(42)
    N = 50
    a = rng.uniform(-1, 1, N)
    b = rng.uniform(2, 5, N)
    c = rng.uniform(-1, 1, N)
    d = rng.uniform(-10, 10, N)
    a[0] = 0.0
    c[-1] = 0.0
    x = thomas_solve(a, b, c, d)
    A = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    x_ref = np.linalg.solve(A, d)
    assert np.allclose(x, x_ref, rtol=1e-12)

def test_does_not_modify_inputs():
    """Thomas solve should not mutate input arrays."""
    N = 10
    a = np.zeros(N)
    b = np.ones(N) * 2
    c = np.ones(N) * -0.5
    d = np.arange(N, dtype=float)
    a_orig, b_orig, c_orig, d_orig = a.copy(), b.copy(), c.copy(), d.copy()
    thomas_solve(a, b, c, d)
    assert np.array_equal(a, a_orig)
    assert np.array_equal(b, b_orig)
    assert np.array_equal(c, c_orig)
    assert np.array_equal(d, d_orig)
