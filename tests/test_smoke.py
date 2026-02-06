# Copyright (c) 2026 Shmuel Link
# SPDX-License-Identifier: MIT

# tests/test_smoke.py
def test_import():
    import fluidflow
    from fluidflow.models.burgers import Burgers1D
    from fluidflow.solvers.time_integrators import RK4
