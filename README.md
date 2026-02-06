# Laminarization Transition in Oscillatory Boundary Layers

Computational study of laminarization transition in sediment-laden oscillatory boundary layers. Produces phase maps and scaling laws describing when turbulence collapses under sediment stratification.

## Quick start

```bash
uv sync
uv run fluidflow-sweep --Re 50 100 --S 1.0 --Lambda 0.1 -N 32 --cycles 2
uv run pytest
```

## Installation

**With uv (recommended):**

```bash
uv sync                    # install project + dependencies
uv sync --group dev        # include dev dependencies (pytest)
```

**With pip:**

```bash
pip install -e ".[dev]"
```

## Running sweeps

The `fluidflow-sweep` command runs parameter sweeps over (Re, S, Lambda) and classifies each case as turbulent or laminar.

**Quick demo** (small grid, few cycles):

```bash
fluidflow-sweep --Re 50 100 --S 1.0 --Lambda 0.1 -N 32 --cycles 2 --outdir results/demo
```

**Production sweep:**

```bash
fluidflow-sweep \
    --Re 50 100 200 400 \
    --S 0.5 1.0 2.0 \
    --Lambda 0.01 0.1 0.5 1.0 \
    -N 128 -H 5.0 --gamma 2.0 \
    --cycles 20 --workers 4 \
    --outdir results/production
```

Each run saves per-case JSON files and a `summary.csv` to the output directory.

### CLI options

| Flag | Description | Default |
|------|-------------|---------|
| `--Re` | Reynolds number values | (required) |
| `--S` | Settling number values | (required) |
| `--Lambda` | Stratification parameter values | (required) |
| `-N` | Grid points | 128 |
| `-H` | Domain height (Stokes-layer thicknesses) | 5.0 |
| `--gamma` | Grid stretching parameter | 2.0 |
| `--Sc_t` | Turbulent Schmidt number | 1.0 |
| `--cycles` | Oscillation cycles | 20 |
| `--workers` | Parallel workers | CPU count |
| `--outdir` | Output directory | `results/` |

## Project structure

```
src/fluidflow/
├── cli.py                 # CLI entry point (fluidflow-sweep)
├── sweep.py               # Parameter sweep driver (build_sweep_grid, run_sweep)
├── io.py                  # JSON serialization for results
├── grid.py                # Stretched grid with tanh clustering near bed
├── diagnostics.py         # Cycle-averaged diagnostics and regime classification
├── models/
│   ├── base.py            # PDEModel ABC: subclasses implement rhs(u, t)
│   ├── oscillatory_bl.py  # Coupled momentum + sediment boundary layer model
│   ├── closures.py        # Turbulence closure nu_t(C) with Richardson damping
│   └── burgers.py         # Burgers equation (test scaffold)
└── solvers/
    ├── operators.py       # Spatial derivatives (FD on stretched grid)
    ├── time_integrators.py # RK4 integrator
    ├── imex.py            # IMEX time stepping (implicit diffusion)
    └── tridiagonal.py     # Thomas algorithm for implicit solves
```

## Parameters

| Symbol | Name | Physical meaning |
|--------|------|-----------------|
| Re | Reynolds number | Ratio of inertial to viscous forces: U_0 * delta / nu |
| S | Settling number | Ratio of settling velocity to flow velocity: w_s / U_0 |
| Lambda | Stratification parameter | Strength of buoyancy effects: g * beta * C_0 * delta / U_0^2 |
| N | Grid points | Vertical resolution |
| H | Domain height | In units of Stokes-layer thickness delta |
| gamma | Stretching | Grid clustering near bed (0 = uniform) |
| Sc_t | Turbulent Schmidt number | Ratio of momentum to mass diffusivity: nu_t / D_t |

## Building the paper

```bash
bash paper/build.sh        # requires pdflatex + bibtex (TeX Live)
```

Figures are regenerated from simulation data with:

```bash
python paper/figures/generate_figures.py
```

## Tests

```bash
uv run pytest              # run all tests
uv run pytest -v           # verbose output
```

## Data and Code Availability

All simulation code and data are publicly available at:
https://github.com/slink/laminarization-transition-study

## License

MIT
