"""
scm_environment_model_comparison.py
=====================================
Framework SCM – Motor de Velos
Comparative analysis of galaxy rotation curve models.

This module performs a systematic comparison of multiple dark-matter and
modified-gravity models fitted to one or more observed rotation curves.
It produces:
  - Per-galaxy ranking tables (BIC, AIC, reduced χ²)
  - Parameter posterior summaries (bootstrap uncertainties)
  - Stacked comparison plots
  - A CSV summary exported to the output directory

Models compared
---------------
  1. Keplerian    – point-mass reference
  2. Isothermal   – pseudo-isothermal sphere (core)
  3. NFW          – Navarro–Frenk–White (cusp)
  4. Burkert      – Burkert cored halo
  5. MOND         – deep-MOND baryonic prediction

Usage:
    python scm_environment_model_comparison.py --catalog data/catalog.csv \
                                               --output comparisons/

    The catalog CSV must contain columns:
        galaxy, radius_kpc, velocity_kms, velocity_err_kms
"""

import argparse
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    warnings.warn("matplotlib not available – plots will be skipped.")

try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available – model fitting will be limited.")

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
G_SI = 6.674e-11
KPC_TO_M = 3.0857e19
KM_TO_M = 1.0e3
M_SUN = 1.989e30

# ---------------------------------------------------------------------------
# Rotation curve models (same definitions as generate_scm_framework_report.py)
# ---------------------------------------------------------------------------

def v_isothermal(r: np.ndarray, v_inf: float, r_c: float) -> np.ndarray:
    """Pseudo-isothermal sphere velocity [km/s]."""
    return v_inf * np.sqrt(1.0 - (r_c / r) * np.arctan(r / r_c))


def v_nfw(r: np.ndarray, v200: float, c: float) -> np.ndarray:
    """NFW halo velocity [km/s]."""
    x = c * r / (r[-1] if len(r) > 0 else 1.0)
    f_c = np.log(1.0 + c) - c / (1.0 + c)
    return v200 * np.sqrt((np.log(1.0 + x) - x / (1.0 + x)) / (x * f_c))


def v_burkert(r: np.ndarray, v0: float, r0: float) -> np.ndarray:
    """Burkert halo velocity [km/s].

    Derived from ρ(r) = ρ₀ / [(1 + r/r₀)(1 + (r/r₀)²)].
    """
    x = r / r0
    f = 0.5 * np.log(1.0 + x**2) + np.log(1.0 + x) - np.arctan(x)
    return v0 * np.sqrt(f / x)


def v_mond(r: np.ndarray, M_bar: float, a0: float = 1.2e-10) -> np.ndarray:
    """Deep-MOND flat-curve prediction [km/s]."""
    v4 = G_SI * (M_bar * M_SUN) * a0
    return (v4 ** 0.25 / KM_TO_M) * np.ones_like(r)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Dict] = {
    "Isothermal": {
        "func": v_isothermal,
        "param_names": ["v_inf [km/s]", "r_c [kpc]"],
        "p0_scale": lambda v_flat, r_half: [v_flat, r_half],
        "bounds_lo": [0.0, 0.01],
        "bounds_hi": [2000.0, 200.0],
    },
    "NFW": {
        "func": v_nfw,
        "param_names": ["v200 [km/s]", "c"],
        "p0_scale": lambda v_flat, r_half: [v_flat * 1.5, 10.0],
        "bounds_lo": [0.0, 0.5],
        "bounds_hi": [3000.0, 200.0],
    },
    "Burkert": {
        "func": v_burkert,
        "param_names": ["v0 [km/s]", "r0 [kpc]"],
        "p0_scale": lambda v_flat, r_half: [v_flat, r_half],
        "bounds_lo": [0.0, 0.01],
        "bounds_hi": [2000.0, 200.0],
    },
    "MOND": {
        "func": v_mond,
        "param_names": ["M_bar [M_sun]"],
        "p0_scale": lambda v_flat, r_half: [1e10],
        "bounds_lo": [1e6],
        "bounds_hi": [1e13],
    },
}

# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

FitResult = Dict  # keys: popt, pcov, chi2_red, aic, bic, v_model


def fit_single_model(
    model_name: str,
    r: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
) -> FitResult:
    """Fit one model to a rotation curve and return quality metrics."""
    spec = MODEL_REGISTRY[model_name]
    func = spec["func"]
    n = len(r)

    v_flat = float(np.median(v_obs[-max(3, n // 5):]))
    r_half = float(r[np.searchsorted(r, r.max() / 2)])
    p0 = spec["p0_scale"](v_flat, r_half)
    bounds = (spec["bounds_lo"], spec["bounds_hi"])

    if not HAS_SCIPY:
        return {
            "popt": None, "pcov": None,
            "chi2_red": np.nan, "aic": np.nan, "bic": np.nan,
            "v_model": np.full_like(r, np.nan),
        }

    try:
        popt, pcov = curve_fit(
            func, r, v_obs, p0=p0, sigma=v_err, absolute_sigma=True,
            bounds=bounds, maxfev=15_000,
        )
        v_model = func(r, *popt)
        residuals = (v_obs - v_model) / v_err
        chi2_val = float(np.sum(residuals ** 2))
        k = len(popt)
        dof = max(n - k, 1)
        return {
            "popt": popt,
            "pcov": pcov,
            "chi2_red": chi2_val / dof,
            "aic": chi2_val + 2.0 * k,
            "bic": chi2_val + k * np.log(n),
            "v_model": v_model,
        }
    except Exception as exc:
        warnings.warn(f"[{model_name}] fitting failed: {exc}")
        return {
            "popt": None, "pcov": None,
            "chi2_red": np.nan, "aic": np.nan, "bic": np.nan,
            "v_model": np.full_like(r, np.nan),
        }


def fit_all_models(
    r: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
) -> Dict[str, FitResult]:
    """Fit all registered models and return results keyed by model name."""
    return {name: fit_single_model(name, r, v_obs, v_err) for name in MODEL_REGISTRY}


# ---------------------------------------------------------------------------
# Bootstrap uncertainty estimation
# ---------------------------------------------------------------------------

def bootstrap_parameters(
    model_name: str,
    r: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    n_boot: int = 200,
    seed: int = 0,
) -> Optional[np.ndarray]:
    """Return bootstrap parameter samples (n_boot × n_params) or None on failure."""
    if not HAS_SCIPY:
        return None

    spec = MODEL_REGISTRY[model_name]
    func = spec["func"]
    rng = np.random.default_rng(seed)
    n = len(r)
    v_flat = float(np.median(v_obs[-max(3, n // 5):]))
    r_half = float(r[np.searchsorted(r, r.max() / 2)])
    p0 = spec["p0_scale"](v_flat, r_half)
    bounds = (spec["bounds_lo"], spec["bounds_hi"])

    samples: List[np.ndarray] = []
    for _ in range(n_boot):
        v_boot = v_obs + rng.normal(0.0, v_err)
        try:
            popt, _ = curve_fit(
                func, r, v_boot, p0=p0, sigma=v_err, absolute_sigma=True,
                bounds=bounds, maxfev=5_000,
            )
            samples.append(popt)
        except Exception:
            pass

    return np.array(samples) if samples else None


# ---------------------------------------------------------------------------
# Data loading / synthetic generation
# ---------------------------------------------------------------------------

def load_catalog(path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load a multi-galaxy catalog CSV.

    Columns expected: galaxy, radius_kpc, velocity_kms, velocity_err_kms
    Returns a dict {galaxy_name: (r, v, v_err)}.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Catalog not found: {path}")

    catalog: Dict[str, List] = {}
    with open(path, encoding="utf-8") as fh:
        header = next(fh).strip().split(",")
        idx = {col.strip(): i for i, col in enumerate(header)}
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            gal = parts[idx["galaxy"]].strip()
            r_val = float(parts[idx["radius_kpc"]])
            v_val = float(parts[idx["velocity_kms"]])
            ve_val = float(parts[idx.get("velocity_err_kms", -1)]) if "velocity_err_kms" in idx else 5.0
            catalog.setdefault(gal, [[], [], []])
            catalog[gal][0].append(r_val)
            catalog[gal][1].append(v_val)
            catalog[gal][2].append(ve_val)

    return {g: (np.array(d[0]), np.array(d[1]), np.array(d[2]))
            for g, d in catalog.items()}


def generate_synthetic_catalog(
    n_galaxies: int = 3, seed: int = 7
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate synthetic galaxy rotation curves for testing."""
    rng = np.random.default_rng(seed)
    catalog: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    v_flats = rng.uniform(120, 280, n_galaxies)
    r_cs = rng.uniform(1.5, 6.0, n_galaxies)
    for i in range(n_galaxies):
        name = f"Galaxy_{i + 1:03d}"
        r = np.linspace(0.5, 30.0, 40)
        v_true = v_isothermal(r, v_flats[i], r_cs[i])
        noise = rng.uniform(5.0, 12.0)
        v_obs = v_true + rng.normal(0, noise, len(r))
        v_err = noise * np.ones(len(r))
        catalog[name] = (r, v_obs, v_err)
    return catalog


# ---------------------------------------------------------------------------
# Comparison table helpers
# ---------------------------------------------------------------------------

def rank_models(fit_results: Dict[str, FitResult], criterion: str = "bic") -> List[Tuple[str, float]]:
    """Return list of (model_name, metric_value) sorted ascending by criterion."""
    pairs = [
        (name, res[criterion])
        for name, res in fit_results.items()
        if not np.isnan(res[criterion])
    ]
    return sorted(pairs, key=lambda x: x[1])


def delta_bic(fit_results: Dict[str, FitResult]) -> Dict[str, float]:
    """Return ΔBIC = BIC_model − BIC_best for each model."""
    ranked = rank_models(fit_results, "bic")
    if not ranked:
        return {name: np.nan for name in fit_results}
    best_bic = ranked[0][1]
    return {name: res["bic"] - best_bic for name, res in fit_results.items()}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison_grid(
    catalog: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    all_fits: Dict[str, Dict[str, FitResult]],
    output_dir: str,
) -> None:
    """Save a multi-panel rotation-curve comparison figure per galaxy."""
    if not HAS_MPL:
        return

    n_models = len(MODEL_REGISTRY)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for galaxy, (r, v_obs, v_err) in catalog.items():
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax_v, ax_r = axes

        # Velocity panel
        ax_v.errorbar(r, v_obs, yerr=v_err, fmt="o", color="black",
                      label="Observed", zorder=5, markersize=3)
        fits = all_fits.get(galaxy, {})
        for (name, res), color in zip(fits.items(), colors):
            v_m = res["v_model"]
            if not np.all(np.isnan(v_m)):
                chi2_r = res["chi2_red"]
                label = f"{name} (χ²_r={chi2_r:.2f})" if not np.isnan(chi2_r) else name
                ax_v.plot(r, v_m, label=label, color=color, linewidth=1.8)
        ax_v.set_xlabel("Radius [kpc]")
        ax_v.set_ylabel("v [km/s]")
        ax_v.set_title(f"{galaxy} – Model comparison")
        ax_v.legend(fontsize=7)
        ax_v.grid(True, alpha=0.3)

        # ΔBIC bar chart
        d_bic = delta_bic(fits)
        names = list(d_bic.keys())
        vals = [d_bic[n] if not np.isnan(d_bic.get(n, np.nan)) else 0 for n in names]
        bar_colors = [colors[i % len(colors)] for i in range(len(names))]
        ax_r.barh(names, vals, color=bar_colors, alpha=0.8)
        ax_r.axvline(0, color="black", linewidth=0.8)
        ax_r.axvline(10, color="grey", linewidth=0.8, linestyle="--", label="ΔBIC=10")
        ax_r.set_xlabel("ΔBIC (lower = better)")
        ax_r.set_title("Bayesian model selection")
        ax_r.legend(fontsize=8)
        ax_r.grid(True, alpha=0.3, axis="x")

        fig.tight_layout()
        out_path = os.path.join(output_dir, f"{galaxy}_model_comparison.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_summary_bic_heatmap(
    galaxies: List[str],
    all_fits: Dict[str, Dict[str, FitResult]],
    output_dir: str,
) -> None:
    """Save a heatmap of ΔBIC values across galaxies × models."""
    if not HAS_MPL:
        return

    model_names = list(MODEL_REGISTRY.keys())
    matrix = np.full((len(galaxies), len(model_names)), np.nan)
    for i, gal in enumerate(galaxies):
        d_bic = delta_bic(all_fits.get(gal, {}))
        for j, mname in enumerate(model_names):
            matrix[i, j] = d_bic.get(mname, np.nan)

    fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 1.5), max(4, len(galaxies) * 0.8)))
    valid = matrix[~np.isnan(matrix)]
    vmax = np.percentile(valid, 95) if len(valid) > 0 else 20
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=30, ha="right")
    ax.set_yticks(range(len(galaxies)))
    ax.set_yticklabels(galaxies)
    ax.set_title("ΔBIC heatmap (lower = better fit)")
    fig.colorbar(im, ax=ax, label="ΔBIC")
    fig.tight_layout()
    out_path = os.path.join(output_dir, "summary_bic_heatmap.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# CSV summary export
# ---------------------------------------------------------------------------

def export_summary_csv(
    all_fits: Dict[str, Dict[str, FitResult]],
    output_dir: str,
) -> str:
    """Write comparison results to CSV and return the path."""
    path = os.path.join(output_dir, "model_comparison_summary.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("galaxy,model,chi2_red,aic,bic,delta_bic\n")
        for galaxy, fits in all_fits.items():
            d_bic = delta_bic(fits)
            for model_name, res in fits.items():
                fh.write(
                    f"{galaxy},{model_name},"
                    f"{res['chi2_red']:.4f},{res['aic']:.2f},"
                    f"{res['bic']:.2f},{d_bic.get(model_name, np.nan):.2f}\n"
                )
    return path


# ---------------------------------------------------------------------------
# Main comparison pipeline
# ---------------------------------------------------------------------------

def run_comparison(
    catalog: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: str = "comparisons",
    bootstrap: bool = False,
    n_boot: int = 200,
) -> Dict[str, Dict[str, FitResult]]:
    """Run the model comparison pipeline over all galaxies in the catalog.

    Parameters
    ----------
    catalog : dict
        {galaxy_name: (r, v_obs, v_err)}
    output_dir : str
    bootstrap : bool
        If True, compute bootstrap parameter uncertainties.
    n_boot : int
        Number of bootstrap iterations (used only when bootstrap=True).

    Returns
    -------
    all_fits : dict  {galaxy: {model: FitResult}}
    """
    os.makedirs(output_dir, exist_ok=True)
    all_fits: Dict[str, Dict[str, FitResult]] = {}

    for galaxy, (r, v_obs, v_err) in catalog.items():
        print(f"\n── {galaxy} ──")
        fits = fit_all_models(r, v_obs, v_err)
        all_fits[galaxy] = fits

        ranked = rank_models(fits, "bic")
        d_bic = delta_bic(fits)
        print(f"  {'Model':<15} {'χ²_red':>8} {'ΔBIC':>8}")
        for name, bic_val in ranked:
            chi2_r = fits[name]["chi2_red"]
            db = d_bic.get(name, np.nan)
            marker = " ◀ best" if db == 0.0 else ""
            print(f"  {name:<15} {chi2_r:>8.3f} {db:>8.2f}{marker}")

        if bootstrap:
            best_name = ranked[0][0] if ranked else None
            if best_name:
                print(f"  Bootstrap [{best_name}] n={n_boot} ...", end=" ", flush=True)
                samples = bootstrap_parameters(best_name, r, v_obs, v_err, n_boot)
                if samples is not None and len(samples) > 10:
                    param_names = MODEL_REGISTRY[best_name]["param_names"]
                    for pname, col in zip(param_names, samples.T):
                        med = np.median(col)
                        lo, hi = np.percentile(col, [16, 84])
                        print(f"\n    {pname}: {med:.3g} [{lo:.3g}, {hi:.3g}]", end="")
                print()

    # Plots
    print("\n── Generating comparison plots ──")
    plot_comparison_grid(catalog, all_fits, output_dir)
    plot_summary_bic_heatmap(list(catalog.keys()), all_fits, output_dir)

    # CSV
    csv_path = export_summary_csv(all_fits, output_dir)
    print(f"  CSV summary: {csv_path}")

    return all_fits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SCM Framework – model comparison across galaxy sample"
    )
    parser.add_argument("--catalog", "-c", default=None,
                        help="Multi-galaxy catalog CSV (galaxy,radius_kpc,velocity_kms,velocity_err_kms). "
                             "If omitted, synthetic data is used.")
    parser.add_argument("--output", "-o", default="comparisons",
                        help="Output directory (default: comparisons/).")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Compute bootstrap parameter uncertainties for the best model.")
    parser.add_argument("--n-boot", type=int, default=200,
                        help="Number of bootstrap iterations (default: 200).")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    if args.catalog:
        try:
            catalog = load_catalog(args.catalog)
        except (FileNotFoundError, KeyError) as exc:
            print(f"ERROR loading catalog: {exc}", file=sys.stderr)
            return 1
    else:
        print("No catalog provided – using synthetic galaxy sample.")
        catalog = generate_synthetic_catalog()

    run_comparison(
        catalog,
        output_dir=args.output,
        bootstrap=args.bootstrap,
        n_boot=args.n_boot,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
