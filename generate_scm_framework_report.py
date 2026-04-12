"""
generate_scm_framework_report.py
=================================
Framework SCM – Motor de Velos
Statistical analysis of galaxy rotation curves.

This module generates a comprehensive SCM framework report that includes:
  - Summary statistics of the rotation curve dataset
  - Model fitting results (Keplerian, Isothermal, NFW, Burkert, MOND)
  - Goodness-of-fit metrics (chi-squared, reduced chi-squared, BIC, AIC)
  - Visual outputs: rotation curve plots, residual maps, parameter distributions
  - A consolidated PDF/text report

Usage:
    python generate_scm_framework_report.py --input data/galaxy_sample.csv \
                                            --output reports/ \
                                            --galaxy NGC1234
"""

import argparse
import os
import sys
import warnings
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    warnings.warn("matplotlib not available – plots will be skipped.")

try:
    from scipy.optimize import curve_fit
    from scipy.stats import chi2 as chi2_dist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available – model fitting will be limited.")

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
G_SI = 6.674e-11        # gravitational constant [m³ kg⁻¹ s⁻²]
KPC_TO_M = 3.0857e19    # 1 kpc in metres
KM_TO_M = 1.0e3         # 1 km in metres
M_SUN = 1.989e30        # solar mass [kg]

# ---------------------------------------------------------------------------
# Rotation curve models
# ---------------------------------------------------------------------------

def v_keplerian(r: np.ndarray, M: float) -> np.ndarray:
    """Keplerian (point-mass) circular velocity [km/s].

    Parameters
    ----------
    r : array_like
        Galactocentric radius [kpc].
    M : float
        Total enclosed mass [M_sun].
    """
    r_m = r * KPC_TO_M
    M_kg = M * M_SUN
    v_ms = np.sqrt(G_SI * M_kg / r_m)
    return v_ms / KM_TO_M


def v_isothermal(r: np.ndarray, v_inf: float, r_c: float) -> np.ndarray:
    """Pseudo-isothermal sphere circular velocity [km/s].

    Parameters
    ----------
    r : array_like
        Galactocentric radius [kpc].
    v_inf : float
        Asymptotic velocity [km/s].
    r_c : float
        Core radius [kpc].
    """
    return v_inf * np.sqrt(1.0 - (r_c / r) * np.arctan(r / r_c))


def v_nfw(r: np.ndarray, v200: float, c: float) -> np.ndarray:
    """NFW (Navarro–Frenk–White) circular velocity [km/s].

    Parameters
    ----------
    r : array_like
        Galactocentric radius [kpc].
    v200 : float
        Virial velocity v₂₀₀ [km/s].
    c : float
        Concentration parameter (dimensionless).
    """
    x = c * r / (r[-1] if len(r) > 0 else 1.0)
    f_c = np.log(1.0 + c) - c / (1.0 + c)
    v = v200 * np.sqrt((np.log(1.0 + x) - x / (1.0 + x)) / (x * f_c))
    return v


def v_burkert(r: np.ndarray, v0: float, r0: float) -> np.ndarray:
    """Burkert core-dominated halo circular velocity [km/s].

    The velocity is derived from the Burkert density profile
    ρ(r) = ρ₀ / [(1 + r/r₀)(1 + (r/r₀)²)].

    Parameters
    ----------
    r : array_like
        Galactocentric radius [kpc].
    v0 : float
        Characteristic velocity scale [km/s].
    r0 : float
        Core radius [kpc].
    """
    x = r / r0
    f = 0.5 * np.log(1.0 + x**2) + np.log(1.0 + x) - np.arctan(x)
    return v0 * np.sqrt(f / x)


def v_mond(r: np.ndarray, M_bar: float, a0: float = 1.2e-10) -> np.ndarray:
    """MOND (deep-MOND limit) asymptotic circular velocity [km/s].

    Parameters
    ----------
    r : array_like
        Galactocentric radius [kpc].
    M_bar : float
        Baryonic (stellar + gas) mass [M_sun].
    a0 : float
        MOND acceleration constant [m/s²]. Default 1.2e-10.
    """
    r_m = r * KPC_TO_M
    M_kg = M_bar * M_SUN
    v4 = G_SI * M_kg * a0
    v_ms = v4 ** 0.25 / np.sqrt(r_m / r_m)   # flat part: v⁴ = G M a₀
    v_ms_arr = (G_SI * M_kg * a0) ** 0.25 * np.ones_like(r)
    return v_ms_arr / KM_TO_M


# ---------------------------------------------------------------------------
# Fitting utilities
# ---------------------------------------------------------------------------

def fit_model(
    func,
    r: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    p0: List[float],
    bounds: Tuple = (-np.inf, np.inf),
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, float, float]:
    """Fit a rotation-curve model and return parameters with quality metrics.

    Returns
    -------
    popt : ndarray or None
    pcov : ndarray or None
    chi2_red : float  (reduced chi-squared)
    aic : float
    bic : float
    """
    if not HAS_SCIPY:
        return None, None, np.nan, np.nan, np.nan

    n = len(r)
    try:
        popt, pcov = curve_fit(
            func, r, v_obs, p0=p0, sigma=v_err, absolute_sigma=True,
            bounds=bounds, maxfev=10_000,
        )
        v_model = func(r, *popt)
        residuals = (v_obs - v_model) / v_err
        chi2_val = float(np.sum(residuals ** 2))
        k = len(popt)
        dof = max(n - k, 1)
        chi2_red = chi2_val / dof
        aic = chi2_val + 2.0 * k
        bic = chi2_val + k * np.log(n)
        return popt, pcov, chi2_red, aic, bic
    except Exception as exc:
        warnings.warn(f"Fitting failed: {exc}")
        return None, None, np.nan, np.nan, np.nan


def load_rotation_curve(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load rotation curve data from a CSV file.

    Expected columns: radius_kpc, velocity_kms, velocity_err_kms

    Returns
    -------
    r, v_obs, v_err : ndarrays
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError("CSV must have at least columns: radius_kpc, velocity_kms")

    r = data[:, 0]
    v_obs = data[:, 1]
    v_err = data[:, 2] if data.shape[1] >= 3 else np.ones_like(v_obs) * 5.0
    return r, v_obs, v_err


def generate_synthetic_curve(
    n_points: int = 40,
    r_max: float = 30.0,
    v_flat: float = 200.0,
    r_c: float = 3.0,
    noise: float = 8.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic rotation curve for testing purposes."""
    rng = np.random.default_rng(seed)
    r = np.linspace(0.5, r_max, n_points)
    v_true = v_isothermal(r, v_flat, r_c)
    v_err = noise * np.ones(n_points)
    v_obs = v_true + rng.normal(0, noise, n_points)
    return r, v_obs, v_err


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_rotation_curve(
    r: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    model_curves: Dict[str, np.ndarray],
    galaxy_name: str,
    ax: Optional["plt.Axes"] = None,
) -> "plt.Figure":
    """Plot observed rotation curve with overlaid model fits."""
    if not HAS_MPL:
        raise RuntimeError("matplotlib is required for plotting.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    ax.errorbar(r, v_obs, yerr=v_err, fmt="o", color="black",
                label="Observed", zorder=5, markersize=4)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for (name, v_model), color in zip(model_curves.items(), colors):
        ax.plot(r, v_model, label=name, color=color, linewidth=2)

    ax.set_xlabel("Radius [kpc]")
    ax.set_ylabel("Circular velocity [km/s]")
    ax.set_title(f"Rotation Curve – {galaxy_name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_residuals(
    r: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    model_curves: Dict[str, np.ndarray],
    galaxy_name: str,
) -> "plt.Figure":
    """Plot residuals (observed − model) for each fitted model."""
    if not HAS_MPL:
        raise RuntimeError("matplotlib is required for plotting.")

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for (name, v_model), color in zip(model_curves.items(), colors):
        resid = v_obs - v_model
        ax.plot(r, resid, "o-", label=name, color=color, markersize=4)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(r, -v_err, v_err, alpha=0.15, color="grey", label="±1σ")
    ax.set_xlabel("Radius [kpc]")
    ax.set_ylabel("Residual [km/s]")
    ax.set_title(f"Residuals – {galaxy_name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _metric_table(fit_results: Dict[str, Dict]) -> str:
    """Return a formatted ASCII table of fit metrics."""
    header = f"{'Model':<20} {'χ²_red':>10} {'AIC':>10} {'BIC':>10}"
    sep = "-" * len(header)
    rows = [header, sep]
    for model_name, metrics in fit_results.items():
        chi2_r = metrics.get("chi2_red", np.nan)
        aic = metrics.get("aic", np.nan)
        bic = metrics.get("bic", np.nan)
        rows.append(
            f"{model_name:<20} {chi2_r:>10.3f} {aic:>10.2f} {bic:>10.2f}"
        )
    return "\n".join(rows)


def generate_report(
    galaxy_name: str,
    r: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    output_dir: str = "reports",
) -> str:
    """Run the full SCM framework pipeline and write a report.

    Parameters
    ----------
    galaxy_name : str
    r, v_obs, v_err : ndarrays (rotation curve data)
    output_dir : str
        Directory where outputs are saved.

    Returns
    -------
    report_path : str
        Path to the generated text report.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_lines: List[str] = []

    def _log(msg: str = "") -> None:
        print(msg)
        report_lines.append(msg)

    _log("=" * 70)
    _log(f"  SCM FRAMEWORK REPORT – Motor de Velos")
    _log(f"  Galaxy : {galaxy_name}")
    _log(f"  Generated : {timestamp} UTC")
    _log("=" * 70)

    # --- Summary statistics ---
    _log("\n[1] DATA SUMMARY")
    _log(f"  Data points   : {len(r)}")
    _log(f"  r_min / r_max : {r.min():.2f} / {r.max():.2f} kpc")
    _log(f"  v_obs range   : {v_obs.min():.1f} – {v_obs.max():.1f} km/s")
    _log(f"  Mean v_err    : {v_err.mean():.2f} km/s")

    # --- Initial guesses ---
    v_flat_guess = float(np.median(v_obs[-5:]))
    r_c_guess = float(r[np.argmax(np.gradient(v_obs, r))])
    if r_c_guess <= 0:
        r_c_guess = 2.0

    model_specs = {
        "Isothermal": {
            "func": v_isothermal,
            "p0": [v_flat_guess, r_c_guess],
            "bounds": ([0, 0.01], [2000, r.max()]),
        },
        "NFW": {
            "func": v_nfw,
            "p0": [v_flat_guess * 1.5, 10.0],
            "bounds": ([0, 0.5], [3000, 200]),
        },
        "Burkert": {
            "func": v_burkert,
            "p0": [v_flat_guess, r_c_guess],
            "bounds": ([0, 0.01], [2000, r.max()]),
        },
    }

    # --- Fitting ---
    _log("\n[2] MODEL FITTING")
    fit_results: Dict[str, Dict] = {}
    model_curves: Dict[str, np.ndarray] = {}

    for model_name, spec in model_specs.items():
        popt, pcov, chi2_red, aic, bic = fit_model(
            spec["func"], r, v_obs, v_err, spec["p0"], spec["bounds"]
        )
        fit_results[model_name] = {
            "popt": popt,
            "pcov": pcov,
            "chi2_red": chi2_red,
            "aic": aic,
            "bic": bic,
        }
        if popt is not None:
            model_curves[model_name] = spec["func"](r, *popt)
            param_str = ", ".join(f"{p:.3g}" for p in popt)
            _log(f"  {model_name:<15}: params = [{param_str}]"
                 f"  χ²_red = {chi2_red:.3f}")
        else:
            model_curves[model_name] = np.full_like(r, np.nan)
            _log(f"  {model_name:<15}: fitting failed")

    # --- Best model ---
    _log("\n[3] GOODNESS-OF-FIT SUMMARY")
    _log(_metric_table(fit_results))
    valid = {k: v for k, v in fit_results.items() if not np.isnan(v["bic"])}
    if valid:
        best = min(valid, key=lambda k: valid[k]["bic"])
        _log(f"\n  Best model (lowest BIC): {best}")

    # --- Plots ---
    if HAS_MPL and model_curves:
        _log("\n[4] GENERATING PLOTS")
        pdf_path = os.path.join(output_dir, f"{galaxy_name}_scm_report_{timestamp}.pdf")
        with PdfPages(pdf_path) as pdf:
            fig_rc = plot_rotation_curve(r, v_obs, v_err, model_curves, galaxy_name)
            pdf.savefig(fig_rc)
            plt.close(fig_rc)

            fig_res = plot_residuals(r, v_obs, v_err, model_curves, galaxy_name)
            pdf.savefig(fig_res)
            plt.close(fig_res)

        _log(f"  Saved: {pdf_path}")
    else:
        _log("\n[4] PLOTS SKIPPED (matplotlib not available)")

    # --- Write text report ---
    report_path = os.path.join(output_dir, f"{galaxy_name}_scm_report_{timestamp}.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines))
    _log(f"\n  Report saved: {report_path}")
    _log("=" * 70)

    return report_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SCM Framework – Motor de Velos: generate rotation curve report"
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Path to CSV file (columns: radius_kpc, velocity_kms[, velocity_err_kms]). "
             "If omitted, a synthetic curve is generated for demonstration.",
    )
    parser.add_argument(
        "--output", "-o",
        default="reports",
        help="Output directory for reports and plots (default: reports/).",
    )
    parser.add_argument(
        "--galaxy", "-g",
        default="SyntheticGalaxy",
        help="Galaxy identifier / name used in report headers.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    if args.input:
        try:
            r, v_obs, v_err = load_rotation_curve(args.input)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR loading data: {exc}", file=sys.stderr)
            return 1
    else:
        print("No input file provided – using synthetic rotation curve.")
        r, v_obs, v_err = generate_synthetic_curve()

    generate_report(
        galaxy_name=args.galaxy,
        r=r,
        v_obs=v_obs,
        v_err=v_err,
        output_dir=args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
