#!/usr/bin/env python
"""Generate all figures for the paper from sweep results.

Usage:
    python paper/figures/generate_figures.py [--format article|aip|both]

Outputs PDF figures to paper/figures/ (article) and/or paper/figures-aip/ (AIP).
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
PAPER_DIR = ROOT / "paper"

# Format-specific configuration for figure sizes and layouts
FORMATS = {
    "article": {
        "output_dir": PAPER_DIR / "figures",
        "regime_diagram": {"figsize": (14, 3.2), "grid": (1, 5)},
        "viscosity_slices": {"figsize": (12, 6.5), "grid": (2, 3)},
        "grid_convergence": {"figsize": (5.5, 4), "legend_loc": "outside"},
        "production_diagnostics": {"figsize": (10, 4)},
        "phase_portraits": {"figsize": (13, 3.2), "grid": (1, 4)},
        "critical_lambda": {"figsize": (5.5, 4)},
        "reentrant_profiles": {"figsize": (12, 4.5)},
        "damping_comparison": {"figsize": (12, 8), "grid": (2, 2)},
    },
    "aip": {
        "output_dir": PAPER_DIR / "figures-aip",
        "regime_diagram": {"figsize": (7.0, 4.5), "grid": (2, 3)},
        "viscosity_slices": {"figsize": (7.0, 5.0), "grid": (2, 3)},
        "grid_convergence": {"figsize": (3.4, 3.0), "legend_loc": "inside"},
        "production_diagnostics": {"figsize": (7.0, 3.0)},
        "phase_portraits": {"figsize": (7.0, 5.5), "grid": (2, 2)},
        "critical_lambda": {"figsize": (3.4, 3.0)},
        "reentrant_profiles": {"figsize": (7.0, 4.0)},
        "damping_comparison": {"figsize": (7.0, 5.5), "grid": (2, 2)},
    },
}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
TURBULENT_COLOR = "#d62728"
LAMINAR_COLOR = "#1f77b4"
PALETTE = {"turbulent": TURBULENT_COLOR, "laminar": LAMINAR_COLOR}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_summary(path):
    """Load a summary CSV into a list of dicts with numeric fields."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "Re": float(row["Re"]),
                "S": float(row["S"]),
                "Lambda": float(row["Lambda"]),
                "regime": row["regime"],
                "converged": row["converged"] == "True",
                "viscosity_ratio": float(row["viscosity_ratio"]),
            })
    return rows


def load_phase2():
    """Load grid convergence data from results/grid-convergence/."""
    records = []
    gc_dir = RESULTS / "grid-convergence"
    if gc_dir.is_dir():
        for Ndir in sorted(gc_dir.iterdir()):
            if not Ndir.is_dir() or not Ndir.name.startswith("N"):
                continue
            N = int(Ndir.name[1:])
            csv_path = Ndir / "summary.csv"
            if csv_path.exists():
                for row in load_summary(csv_path):
                    row["N"] = N
                    records.append(row)

    # Add N=512 production data if available
    n512_dir = RESULTS / "n512-production"
    if n512_dir.is_dir():
        for json_file in sorted(n512_dir.glob("Re*.json")):
            with open(json_file) as f:
                data = json.load(f)
            p = data["params"]
            # Only include cases that match grid convergence parameter combos
            key = (p["Re"], p["S"], p["Lambda"])
            gc_keys = set((r["Re"], r["S"], r["Lambda"]) for r in records)
            if key in gc_keys:
                records.append({
                    "Re": p["Re"],
                    "S": p["S"],
                    "Lambda": p["Lambda"],
                    "regime": data["regime"],
                    "converged": data["converged"],
                    "viscosity_ratio": data["viscosity_ratio"],
                    "N": 512,
                })
    return records


def load_json_results(phase_dir):
    """Load all JSON result files from a phase directory."""
    results = []
    for f in sorted(phase_dir.iterdir()):
        if f.suffix == ".json":
            with open(f) as fp:
                results.append(json.load(fp))
    return results


# ---------------------------------------------------------------------------
# Figure 1: Regime phase diagram (Re vs Lambda, panels by S)
# ---------------------------------------------------------------------------

def fig_regime_diagram(phase1, supplementary, fmt_config, output_dir):
    """Multi-panel regime phase diagram."""
    combined = [r for r in phase1 + supplementary if r["converged"]]

    S_values = sorted(set(r["S"] for r in combined))
    cfg = fmt_config["regime_diagram"]
    nrows, ncols = cfg["grid"]
    figsize = cfg["figsize"]

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for i, (ax, S) in enumerate(zip(axes, S_values)):
        subset = [r for r in combined if r["S"] == S]
        turb = [r for r in subset if r["regime"] == "turbulent"]
        lam = [r for r in subset if r["regime"] == "laminar"]

        ax.scatter([r["Lambda"] for r in turb], [r["Re"] for r in turb],
                   c=TURBULENT_COLOR, marker="s", s=18, alpha=0.7,
                   label="Turbulent", edgecolors="none")
        ax.scatter([r["Lambda"] for r in lam], [r["Re"] for r in lam],
                   c=LAMINAR_COLOR, marker="o", s=18, alpha=0.7,
                   label="Laminar", edgecolors="none")

        ax.set_title(f"$S = {S}$")
        ax.set_xlabel(r"$\Lambda$")
        if i % ncols == 0:
            ax.set_ylabel(r"$Re$")
        ax.set_xscale("symlog", linthresh=0.01)

    # Hide unused axes
    for ax in axes[len(S_values):]:
        ax.set_visible(False)

    handles = [
        mlines.Line2D([], [], color=TURBULENT_COLOR, marker="s",
                       linestyle="None", markersize=5, label="Turbulent"),
        mlines.Line2D([], [], color=LAMINAR_COLOR, marker="o",
                       linestyle="None", markersize=5, label="Laminar"),
    ]
    axes[len(S_values) - 1].legend(handles=handles, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "regime_diagram.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  regime_diagram.pdf")


# ---------------------------------------------------------------------------
# Figure 2: Viscosity ratio vs Lambda slices
# ---------------------------------------------------------------------------

def fig_viscosity_slices(phase1, supplementary, fmt_config, output_dir):
    """Viscosity ratio vs Lambda for selected (Re, S) slices."""
    combined = [r for r in phase1 + supplementary if r["converged"]]

    slices = [
        (300, 0.005), (300, 0.05), (300, 0.1), (300, 0.5),
        (500, 0.01), (500, 0.1),
    ]

    cfg = fmt_config["viscosity_slices"]
    nrows, ncols = cfg["grid"]
    figsize = cfg["figsize"]

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axes = axes.flatten()

    for i, (ax, (Re, S)) in enumerate(zip(axes, slices)):
        subset = sorted(
            [r for r in combined if r["Re"] == Re and r["S"] == S],
            key=lambda r: r["Lambda"],
        )
        if not subset:
            continue

        Lam = [r["Lambda"] for r in subset]
        vr = [r["viscosity_ratio"] for r in subset]
        colors = [PALETTE[r["regime"]] for r in subset]

        ax.scatter(Lam, vr, c=colors, s=22, zorder=3, edgecolors="none")
        ax.plot(Lam, vr, color="0.5", linewidth=0.7, alpha=0.5, zorder=2)
        ax.axhline(10, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(f"$Re = {int(Re)},\\; S = {S}$", fontsize=10)
        ax.set_xlabel(r"$\Lambda$")
        if i % ncols == 0:
            ax.set_ylabel(r"$\langle\nu_t\rangle / \nu$")
        ax.set_ylim(0, max(vr) * 1.15 if vr else 25)

    fig.tight_layout()
    fig.savefig(output_dir / "viscosity_slices.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  viscosity_slices.pdf")


# ---------------------------------------------------------------------------
# Figure 3: Grid convergence (Phase 2)
# ---------------------------------------------------------------------------

def fig_grid_convergence(phase2, fmt_config, output_dir):
    """Viscosity ratio vs grid resolution for Phase 2 cases."""
    cases = {}
    for r in phase2:
        key = (r["Re"], r["S"], r["Lambda"])
        cases.setdefault(key, []).append(r)

    cfg = fmt_config["grid_convergence"]
    figsize = cfg["figsize"]
    legend_loc = cfg["legend_loc"]

    fig, ax = plt.subplots(figsize=figsize)
    markers = ["o", "s", "D", "^", "v", "p"]
    colors = sns.color_palette("colorblind", len(cases))

    for i, (key, records) in enumerate(sorted(cases.items())):
        records.sort(key=lambda r: r["N"])
        Ns = [r["N"] for r in records]
        vrs = [r["viscosity_ratio"] for r in records]
        Re, S, Lam = key
        label = f"$Re={int(Re)},\\,S={S},\\,\\Lambda={Lam}$"
        ax.plot(Ns, vrs, marker=markers[i % len(markers)], color=colors[i],
                markersize=6, linewidth=1.2, label=label)

    ax.axhline(10, color="k", linestyle="--", linewidth=0.8, alpha=0.6,
               label=r"$\langle\nu_t\rangle/\nu = 10$")
    ax.set_xlabel("Grid points $N$")
    ax.set_ylabel(r"$\langle\nu_t\rangle / \nu$")
    all_Ns = sorted(set(r["N"] for r in phase2))
    ax.set_xticks(all_Ns)

    if legend_loc == "outside":
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    else:
        ax.legend(fontsize=6, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_dir / "grid_convergence.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  grid_convergence.pdf")


# ---------------------------------------------------------------------------
# Figure 4: Drag coefficient and kinetic energy (Phase 4)
# ---------------------------------------------------------------------------

def fig_production_diagnostics(phase4_jsons, fmt_config, output_dir):
    """Drag coefficient and kinetic energy from Phase 4 production runs."""
    cfg = fmt_config["production_diagnostics"]
    figsize = cfg["figsize"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    vr, cf, ke, regimes = [], [], [], []
    for r in phase4_jsons:
        vr.append(r["viscosity_ratio"])
        cf.append(r["drag_coefficient"])
        ke.append(r["kinetic_energy"])
        regimes.append(r["regime"])

    colors = [PALETTE[reg] for reg in regimes]

    # (a) Drag coefficient vs viscosity ratio
    ax1.scatter(vr, cf, c=colors, s=40, edgecolors="0.3", linewidths=0.5,
                zorder=3)
    ax1.axvline(10, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_xlabel(r"$\langle\nu_t\rangle / \nu$")
    ax1.set_ylabel(r"$c_f$")
    ax1.set_title("(a) Drag coefficient")

    # (b) Kinetic energy vs viscosity ratio
    ax2.scatter(vr, ke, c=colors, s=40, edgecolors="0.3", linewidths=0.5,
                zorder=3)
    ax2.axvline(10, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_xlabel(r"$\langle\nu_t\rangle / \nu$")
    ax2.set_ylabel(r"$\langle KE \rangle$")
    ax2.set_title("(b) Kinetic energy")

    handles = [
        mlines.Line2D([], [], color=TURBULENT_COLOR, marker="o",
                       linestyle="None", markersize=6, label="Turbulent"),
        mlines.Line2D([], [], color=LAMINAR_COLOR, marker="o",
                       linestyle="None", markersize=6, label="Laminar"),
    ]
    ax2.legend(handles=handles, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "production_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  production_diagnostics.pdf")


# ---------------------------------------------------------------------------
# Figure 5: Phase portraits (Phase 4)
# ---------------------------------------------------------------------------

def fig_phase_portraits(phase4_jsons, fmt_config, output_dir):
    """Phase portraits (u_bed, tau_bed) for selected cases."""
    # Select 4 representative cases: 2 laminar, 2 turbulent
    laminar = [r for r in phase4_jsons if r["regime"] == "laminar"]
    turbulent = [r for r in phase4_jsons if r["regime"] == "turbulent"]

    # Pick cases that show diversity
    selected = []
    # Laminar: one low vr, one near threshold
    laminar.sort(key=lambda r: r["viscosity_ratio"])
    if laminar:
        selected.append(laminar[0])   # Lowest vr (clearly laminar)
    if len(laminar) > 1:
        selected.append(laminar[-1])  # Highest vr (near-threshold laminar)

    # Turbulent: one near threshold, one clearly turbulent
    turbulent.sort(key=lambda r: r["viscosity_ratio"])
    if turbulent:
        selected.append(turbulent[0])   # Lowest vr turbulent
    if len(turbulent) > 1:
        selected.append(turbulent[-1])  # Highest vr turbulent

    cfg = fmt_config["phase_portraits"]
    nrows, ncols = cfg["grid"]
    figsize = cfg["figsize"]

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for i, (ax, r) in enumerate(zip(axes, selected)):
        pp = np.array(r["phase_portrait"])
        u_bed = pp[:, 0]
        tau_bed = pp[:, 1]
        color = PALETTE[r["regime"]]

        ax.plot(u_bed, tau_bed, "-o", color=color, markersize=3,
                linewidth=1.0, alpha=0.8)
        ax.plot(u_bed[0], tau_bed[0], "k^", markersize=6, zorder=5)

        p = r["params"]
        vr = r["viscosity_ratio"]
        ax.set_title(
            f"$Re={int(p['Re'])},\\,S={p['S']},\\,\\Lambda={p['Lambda']:.2f}$"
            f"\n{r['regime']} ($\\nu_t/\\nu={vr:.1f}$)",
            fontsize=9,
        )
        ax.set_xlabel(r"$u_{\mathrm{bed}}$")
        if i % ncols == 0:
            ax.set_ylabel(r"$\tau_{\mathrm{bed}}$")

    # Hide unused axes
    for ax in axes[len(selected):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "phase_portraits.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  phase_portraits.pdf")


# ---------------------------------------------------------------------------
# Figure 6: Critical Lambda_c vs Re for low S
# ---------------------------------------------------------------------------

def fig_critical_lambda(phase1, supplementary, fmt_config, output_dir):
    """Critical Lambda_c as a function of Re for S = 0.005 and S = 0.01."""
    combined = [r for r in phase1 + supplementary if r["converged"]]

    cfg = fmt_config["critical_lambda"]
    figsize = cfg["figsize"]

    fig, ax = plt.subplots(figsize=figsize)
    markers = {"0.005": "o", "0.01": "s"}
    colors_s = {"0.005": sns.color_palette("colorblind")[0],
                "0.01": sns.color_palette("colorblind")[1]}

    for S_val in [0.005, 0.01]:
        subset = [r for r in combined if r["S"] == S_val]
        Re_values = sorted(set(r["Re"] for r in subset))

        Re_crit, Lam_crit = [], []
        for Re in Re_values:
            at_Re = sorted(
                [r for r in subset if r["Re"] == Re],
                key=lambda r: r["Lambda"],
            )
            # Find first transition from turbulent to laminar
            for i in range(len(at_Re) - 1):
                if (at_Re[i]["regime"] == "turbulent" and
                        at_Re[i + 1]["regime"] == "laminar"):
                    lc = 0.5 * (at_Re[i]["Lambda"] + at_Re[i + 1]["Lambda"])
                    Re_crit.append(Re)
                    Lam_crit.append(lc)
                    break

        S_key = str(S_val)
        ax.plot(Re_crit, Lam_crit, marker=markers[S_key],
                color=colors_s[S_key], linewidth=1.2, markersize=6,
                label=f"$S = {S_val}$")

    ax.set_xlabel(r"$Re$")
    ax.set_ylabel(r"$\Lambda_c$")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "critical_lambda.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  critical_lambda.pdf")


# ---------------------------------------------------------------------------
# Figure 7: Re-entrant vertical profiles
# ---------------------------------------------------------------------------

def fig_reentrant_profiles(reentrant_jsons, fmt_config, output_dir):
    """Three-panel figure: C(z), Ri_g(z), nu_t(z) for re-entrant profile cases."""
    # Sort by Lambda for consistent ordering
    reentrant_jsons.sort(key=lambda r: r["params"]["Lambda"])

    styles = [
        {"color": "#d62728", "linestyle": "-"},    # Lambda=0.1: solid red (turbulent)
        {"color": "#1f77b4", "linestyle": "-"},    # Lambda=2.0: solid blue (laminar)
        {"color": "#d62728", "linestyle": "--"},   # Lambda=5.0: dashed red (re-entrant turbulent)
    ]

    cfg = fmt_config["reentrant_profiles"]
    figsize = cfg["figsize"]

    fig, (ax_c, ax_ri, ax_nu) = plt.subplots(1, 3, figsize=figsize, sharey=True)

    for r, style in zip(reentrant_jsons, styles):
        prof = r["profiles"]
        z = np.array(prof["z"])
        C = np.array(prof["C"])
        Ri_g = np.array(prof["Ri_g"])
        nu_t = np.array(prof["nu_t"])
        lam = r["params"]["Lambda"]
        label = f"$\\Lambda = {lam}$"

        ax_c.plot(C, z, label=label, **style, linewidth=1.5)
        ax_ri.plot(Ri_g, z, label=label, **style, linewidth=1.5)
        ax_nu.plot(nu_t, z, label=label, **style, linewidth=1.5)

    # Ri_c reference line
    ax_ri.axvline(0.25, color="k", linestyle=":", linewidth=0.8, alpha=0.7,
                  label="$Ri_c = 0.25$")

    ax_c.set_xlabel(r"$\langle C \rangle$")
    ax_c.set_ylabel(r"$z / \delta$")
    ax_c.set_title("(a) Concentration")
    ax_c.legend(fontsize=8)

    ax_ri.set_xlabel(r"$Ri_g$")
    ax_ri.set_title("(b) Gradient Richardson number")
    ax_ri.legend(fontsize=8)

    ax_nu.set_xlabel(r"$\nu_t$")
    ax_nu.set_title("(c) Turbulent viscosity")
    ax_nu.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "reentrant_profiles.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  reentrant_profiles.pdf")


# ---------------------------------------------------------------------------
# Figure 8: Damping function comparison (linear vs exponential)
# ---------------------------------------------------------------------------

def fig_damping_comparison(linear, exponential, fmt_config, output_dir):
    """2x2 panel comparing linear vs exponential damping at S=0.005 and S=0.01."""
    cfg = fmt_config["damping_comparison"]
    nrows, ncols = cfg["grid"]
    figsize = cfg["figsize"]

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    panels = [
        (0.005, "linear", linear),
        (0.005, "exponential", exponential),
        (0.01, "linear", linear),
        (0.01, "exponential", exponential),
    ]
    titles = [
        r"$S = 0.005$, Linear",
        r"$S = 0.005$, Exponential",
        r"$S = 0.01$, Linear",
        r"$S = 0.01$, Exponential",
    ]

    for ax, (S_val, damping_type, data), title in zip(axes, panels, titles):
        subset = [r for r in data if r["converged"] and r["S"] == S_val]
        turb = [r for r in subset if r["regime"] == "turbulent"]
        lam = [r for r in subset if r["regime"] == "laminar"]

        ax.scatter([r["Lambda"] for r in turb], [r["Re"] for r in turb],
                   c=TURBULENT_COLOR, marker="s", s=18, alpha=0.7,
                   label="Turbulent", edgecolors="none")
        ax.scatter([r["Lambda"] for r in lam], [r["Re"] for r in lam],
                   c=LAMINAR_COLOR, marker="o", s=18, alpha=0.7,
                   label="Laminar", edgecolors="none")

        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$\Lambda$")
        ax.set_ylabel(r"$Re$")
        ax.set_xscale("symlog", linthresh=0.01)

    handles = [
        mlines.Line2D([], [], color=TURBULENT_COLOR, marker="s",
                       linestyle="None", markersize=5, label="Turbulent"),
        mlines.Line2D([], [], color=LAMINAR_COLOR, marker="o",
                       linestyle="None", markersize=5, label="Laminar"),
    ]
    axes[-1].legend(handles=handles, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "damping_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  damping_comparison.pdf")


# ---------------------------------------------------------------------------
# Table data: Phase 4 production diagnostics (write as CSV for LaTeX)
# ---------------------------------------------------------------------------

def write_phase4_table(phase4_jsons, output_dir):
    """Write Phase 4 diagnostics to a CSV for LaTeX table."""
    rows = []
    for r in phase4_jsons:
        p = r["params"]
        rows.append({
            "Re": int(p["Re"]),
            "S": p["S"],
            "Lambda": p["Lambda"],
            "regime": r["regime"],
            "viscosity_ratio": r["viscosity_ratio"],
            "drag_coefficient": r["drag_coefficient"],
            "kinetic_energy": r["kinetic_energy"],
        })
    rows.sort(key=lambda r: (r["Re"], r["S"], r["Lambda"]))

    path = output_dir / "phase4_table.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  phase4_table.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_figures_for_format(fmt_name, fmt_config, phase1, supplementary,
                                 phase2, phase4_jsons, reentrant_jsons,
                                 exponential):
    """Generate all figures for a given format."""
    output_dir = fmt_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {fmt_name} figures to {output_dir.name}/...")
    fig_regime_diagram(phase1, supplementary, fmt_config, output_dir)
    fig_viscosity_slices(phase1, supplementary, fmt_config, output_dir)
    fig_grid_convergence(phase2, fmt_config, output_dir)
    fig_production_diagnostics(phase4_jsons, fmt_config, output_dir)
    fig_phase_portraits(phase4_jsons, fmt_config, output_dir)
    fig_critical_lambda(phase1, supplementary, fmt_config, output_dir)
    if reentrant_jsons:
        fig_reentrant_profiles(reentrant_jsons, fmt_config, output_dir)
    if exponential:
        fig_damping_comparison(phase1, exponential, fmt_config, output_dir)
    write_phase4_table(phase4_jsons, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Generate figures for the paper in article and/or AIP format."
    )
    parser.add_argument(
        "--format",
        choices=["article", "aip", "both"],
        default="both",
        help="Output format: article (single-column), aip (two-column), or both (default).",
    )
    args = parser.parse_args()

    print("Loading data...")
    phase1 = load_summary(RESULTS / "phase1-n256" / "summary.csv")
    phase2 = load_phase2()

    # Supplementary data: reentrant-expansion + targeted-s001
    supplementary = []
    for subdir in ["reentrant-expansion", "targeted-s001"]:
        csv_path = RESULTS / subdir / "summary.csv"
        if csv_path.exists():
            supplementary.extend(load_summary(csv_path))

    # Production diagnostics (replaces archive/phase4-production-revised)
    prod_dir = RESULTS / "production-n256"
    phase4_jsons = load_json_results(prod_dir) if prod_dir.is_dir() else []

    # Re-entrant vertical profiles (replaces archive/reentrant_profiles)
    reentrant_dir = RESULTS / "reentrant-profiles"
    reentrant_jsons = load_json_results(reentrant_dir) if reentrant_dir.is_dir() else []

    exponential_summary = RESULTS / "exponential-n256" / "summary.csv"
    exponential = load_summary(exponential_summary) if exponential_summary.exists() else []

    print(f"  Phase 1 N=256 (linear): {len(phase1)} cases")
    print(f"  Phase 2 (grid convergence): {len(phase2)} records")
    print(f"  Supplementary (reentrant + targeted): {len(supplementary)} cases")
    print(f"  Production diagnostics: {len(phase4_jsons)} JSON files")
    print(f"  Re-entrant profiles: {len(reentrant_jsons)} JSON files")
    print(f"  Exponential N=256: {len(exponential)} cases")

    formats_to_generate = (
        ["article", "aip"] if args.format == "both" else [args.format]
    )

    for fmt_name in formats_to_generate:
        generate_figures_for_format(
            fmt_name, FORMATS[fmt_name], phase1, supplementary, phase2,
            phase4_jsons, reentrant_jsons, exponential
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
