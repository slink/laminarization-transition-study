# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

import numpy as np
from numba import njit


@njit(cache=True)
def thomas_solve(a, b, c, d):
    """Solve tridiagonal system Ax = d using the Thomas algorithm.

    Args:
        a: lower diagonal, length N. a[0] is unused.
        b: main diagonal, length N.
        c: upper diagonal, length N. c[-1] is unused.
        d: right-hand side, length N.

    Returns:
        x: solution, length N.
    """
    N = len(b)
    cp = np.empty(N)
    dp = np.empty(N)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, N):
        m = a[i] * cp[i - 1]
        cp[i] = c[i] / (b[i] - m)
        dp[i] = (d[i] - a[i] * dp[i - 1]) / (b[i] - m)
    x = np.empty(N)
    x[-1] = dp[-1]
    for i in range(N - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x
