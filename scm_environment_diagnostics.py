"""
scm_environment_diagnostics.py
================================
Framework SCM – Motor de Velos
Diagnostics and data-quality checks for galaxy rotation curve environments.

This module provides tools to assess the health and reliability of a galaxy
rotation curve dataset before model fitting.  Checks performed include:

  1. Data completeness  – missing values, duplicate radii, empty datasets
  2. Sampling quality   – radial coverage, point spacing uniformity
  3. Velocity sanity    – negative velocities, outlier detection (σ-clipping)
  4. Error sanity       – zero/negative errors, heteroscedasticity flag
  5. S/N assessment     – per-point and median signal-to-noise ratios
  6. Flatness test      – whether the outer curve is consistent with a flat trend
  7. Resolution check   – inner-slope resolution (detects beam smearing proxy)
  8. Environment report – summary printed to stdout and saved as text/CSV

Usage:
    python scm_environment_diagnostics.py --input data/galaxy_sample.csv \
                                          --output diagnostics/ \
                                          --galaxy NGC1234

    Or point to a multi-galaxy catalog:
    python scm_environment_diagnostics.py --catalog data/catalog.csv \
                                          --output diagnostics/
"""

import argparse
import os
import sys
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    warnings.warn("matplotlib not available – diagnostic plots will be skipped.")

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Diagnostic result container
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticResult:
    """Holds all diagnostic flags and metrics for one galaxy."""

    galaxy: str = "unknown"

    # Completeness
    n_points: int = 0
    n_missing: int = 0
    n_duplicate_radii: int = 0

    # Sampling
    r_min_kpc: float = np.nan
    r_max_kpc: float = np.nan
    r_coverage_kpc: float = np.nan
    median_spacing_kpc: float = np.nan
    spacing_cv: float = np.nan          # coefficient of variation of spacing

    # Velocity sanity
    n_negative_v: int = 0
    n_velocity_outliers: int = 0
    v_median_kms: float = np.nan
    v_mad_kms: float = np.nan           # median absolute deviation

    # Error sanity
    n_zero_errors: int = 0
    n_negative_errors: int = 0
    median_err_kms: float = np.nan
    heteroscedastic: bool = False       # True if error spread is large

    # S/N
    median_snr: float = np.nan
    n_low_snr: int = 0                  # points with SNR < 3

    # Flatness (outer rotation curve)
    outer_slope_kms_per_kpc: float = np.nan
    outer_slope_significant: bool = False  # True = rising/falling tail

    # Inner resolution
    inner_slope_kms_per_kpc: float = np.nan
    n_inner_points: int = 0

    # Overall status
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    passed: bool = True


# ---------------------------------------------------------------------------
# Individual diagnostic checks
# ---------------------------------------------------------------------------

def check_completeness(
    r: np.ndarray,
    v: np.ndarray,
    v_err: np.ndarray,
    result: DiagnosticResult,
) -> None:
    """Check for missing values and duplicate radii."""
    n_total = len(r)
    result.n_points = n_total

    if n_total == 0:
        result.errors.append("Dataset is empty.")
        result.passed = False
        return

    # NaN/inf check
    bad = ~(np.isfinite(r) & np.isfinite(v) & np.isfinite(v_err))
    result.n_missing = int(np.sum(bad))
    if result.n_missing > 0:
        pct = 100.0 * result.n_missing / n_total
        result.warnings.append(
            f"{result.n_missing} points ({pct:.1f}%) have non-finite values."
        )

    # Duplicate radii
    _, counts = np.unique(np.round(r, decimals=4), return_counts=True)
    result.n_duplicate_radii = int(np.sum(counts > 1))
    if result.n_duplicate_radii > 0:
        result.warnings.append(
            f"{result.n_duplicate_radii} duplicate radius values detected."
        )


def check_sampling(r: np.ndarray, result: DiagnosticResult) -> None:
    """Assess radial coverage and point spacing uniformity."""
    if len(r) < 2:
        result.errors.append("Too few data points (< 2) for sampling analysis.")
        result.passed = False
        return

    r_sorted = np.sort(r[np.isfinite(r)])
    result.r_min_kpc = float(r_sorted[0])
    result.r_max_kpc = float(r_sorted[-1])
    result.r_coverage_kpc = result.r_max_kpc - result.r_min_kpc

    spacing = np.diff(r_sorted)
    result.median_spacing_kpc = float(np.median(spacing))
    if result.median_spacing_kpc > 0:
        result.spacing_cv = float(np.std(spacing) / result.median_spacing_kpc)
    else:
        result.spacing_cv = np.nan

    if result.r_coverage_kpc < 5.0:
        result.warnings.append(
            f"Radial coverage is small ({result.r_coverage_kpc:.1f} kpc < 5 kpc)."
        )
    if result.spacing_cv > 1.5:
        result.warnings.append(
            f"Highly non-uniform radial spacing (CV = {result.spacing_cv:.2f})."
        )
    if len(r_sorted) < 10:
        result.warnings.append(f"Only {len(r_sorted)} data points – fits may be unreliable.")


def check_velocity_sanity(
    v: np.ndarray,
    result: DiagnosticResult,
    sigma_clip: float = 4.0,
) -> None:
    """Check for negative velocities and statistical outliers."""
    v_finite = v[np.isfinite(v)]
    if len(v_finite) == 0:
        return

    result.n_negative_v = int(np.sum(v_finite < 0))
    if result.n_negative_v > 0:
        result.warnings.append(
            f"{result.n_negative_v} negative velocity value(s) detected."
        )

    result.v_median_kms = float(np.median(v_finite))
    result.v_mad_kms = float(np.median(np.abs(v_finite - result.v_median_kms)))
    mad_sigma = result.v_mad_kms * 1.4826  # robust σ estimate
    if mad_sigma > 0:
        outlier_mask = np.abs(v_finite - result.v_median_kms) > sigma_clip * mad_sigma
        result.n_velocity_outliers = int(np.sum(outlier_mask))
        if result.n_velocity_outliers > 0:
            result.warnings.append(
                f"{result.n_velocity_outliers} velocity outlier(s) > {sigma_clip}σ (MAD-based)."
            )


def check_errors(
    v: np.ndarray,
    v_err: np.ndarray,
    result: DiagnosticResult,
) -> None:
    """Validate uncertainty values."""
    v_err_finite = v_err[np.isfinite(v_err)]
    if len(v_err_finite) == 0:
        return

    result.n_zero_errors = int(np.sum(v_err_finite == 0.0))
    result.n_negative_errors = int(np.sum(v_err_finite < 0.0))
    if result.n_zero_errors > 0:
        result.warnings.append(f"{result.n_zero_errors} zero-error point(s) detected.")
    if result.n_negative_errors > 0:
        result.errors.append(f"{result.n_negative_errors} negative error(s) detected.")
        result.passed = False

    result.median_err_kms = float(np.median(v_err_finite))

    # Heteroscedasticity: flag if error CV > 1.0
    if result.median_err_kms > 0:
        err_cv = float(np.std(v_err_finite) / result.median_err_kms)
        result.heteroscedastic = err_cv > 1.0
        if result.heteroscedastic:
            result.warnings.append(
                f"High error heteroscedasticity (CV = {err_cv:.2f}); "
                "consider per-point weighting in fits."
            )


def check_snr(
    v: np.ndarray,
    v_err: np.ndarray,
    result: DiagnosticResult,
    snr_threshold: float = 3.0,
) -> None:
    """Compute per-point SNR and flag low-quality points."""
    mask = np.isfinite(v) & np.isfinite(v_err) & (v_err > 0)
    if not np.any(mask):
        return

    snr = np.abs(v[mask]) / v_err[mask]
    result.median_snr = float(np.median(snr))
    result.n_low_snr = int(np.sum(snr < snr_threshold))
    if result.n_low_snr > 0:
        pct = 100.0 * result.n_low_snr / np.sum(mask)
        result.warnings.append(
            f"{result.n_low_snr} point(s) ({pct:.1f}%) with SNR < {snr_threshold:.1f}."
        )


def check_flatness(
    r: np.ndarray,
    v: np.ndarray,
    v_err: np.ndarray,
    result: DiagnosticResult,
    outer_fraction: float = 0.3,
    slope_threshold_kms_per_kpc: float = 3.0,
) -> None:
    """Test whether the outer rotation curve is statistically flat.

    A significant non-zero slope in the outer region may indicate a rising
    or declining curve – physically important for dark matter content estimation.
    """
    mask = np.isfinite(r) & np.isfinite(v) & np.isfinite(v_err) & (v_err > 0)
    r_ok, v_ok, ve_ok = r[mask], v[mask], v_err[mask]
    if len(r_ok) < 4:
        return

    n_outer = max(3, int(np.ceil(len(r_ok) * outer_fraction)))
    idx_outer = np.argsort(r_ok)[-n_outer:]
    r_out, v_out, ve_out = r_ok[idx_outer], v_ok[idx_outer], ve_ok[idx_outer]

    if HAS_SCIPY:
        res = scipy_stats.linregress(r_out, v_out)
        slope = res.slope
        slope_err = res.stderr
        result.outer_slope_kms_per_kpc = float(slope)
        # Significant if |slope| > threshold OR |slope| > 2*stderr
        sig1 = abs(slope) > slope_threshold_kms_per_kpc
        sig2 = slope_err > 0 and abs(slope) > 2.0 * slope_err
        result.outer_slope_significant = bool(sig1 or sig2)
    else:
        # Simple numpy polyfit fallback
        coeffs = np.polyfit(r_out, v_out, 1)
        result.outer_slope_kms_per_kpc = float(coeffs[0])
        result.outer_slope_significant = abs(coeffs[0]) > slope_threshold_kms_per_kpc

    if result.outer_slope_significant:
        direction = "rising" if result.outer_slope_kms_per_kpc > 0 else "declining"
        result.warnings.append(
            f"Outer rotation curve appears {direction} "
            f"(slope = {result.outer_slope_kms_per_kpc:.2f} km/s/kpc)."
        )


def check_inner_resolution(
    r: np.ndarray,
    v: np.ndarray,
    result: DiagnosticResult,
    inner_fraction: float = 0.15,
) -> None:
    """Estimate the inner-slope gradient as a proxy for beam-smearing severity."""
    mask = np.isfinite(r) & np.isfinite(v)
    r_ok, v_ok = r[mask], v[mask]
    if len(r_ok) < 4:
        return

    n_inner = max(3, int(np.ceil(len(r_ok) * inner_fraction)))
    idx_inner = np.argsort(r_ok)[:n_inner]
    r_in, v_in = r_ok[idx_inner], v_ok[idx_inner]
    result.n_inner_points = len(r_in)

    if len(r_in) >= 2:
        coeffs = np.polyfit(r_in, v_in, 1)
        result.inner_slope_kms_per_kpc = float(coeffs[0])
        if result.inner_slope_kms_per_kpc < 0:
            result.warnings.append(
                "Negative inner slope detected – possible beam smearing or data artefact."
            )


# ---------------------------------------------------------------------------
# Full diagnostics pipeline
# ---------------------------------------------------------------------------

def run_diagnostics(
    galaxy: str,
    r: np.ndarray,
    v: np.ndarray,
    v_err: np.ndarray,
) -> DiagnosticResult:
    """Run all diagnostic checks and return a DiagnosticResult."""
    result = DiagnosticResult(galaxy=galaxy)

    check_completeness(r, v, v_err, result)
    if not result.passed and result.n_points == 0:
        return result

    # Use finite-only arrays for subsequent checks
    finite_mask = np.isfinite(r) & np.isfinite(v) & np.isfinite(v_err)
    r_f, v_f, ve_f = r[finite_mask], v[finite_mask], v_err[finite_mask]

    check_sampling(r_f, result)
    check_velocity_sanity(v_f, result)
    check_errors(v_f, ve_f, result)
    check_snr(v_f, ve_f, result)
    check_flatness(r_f, v_f, ve_f, result)
    check_inner_resolution(r_f, v_f, result)

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_diagnostic_report(result: DiagnosticResult) -> None:
    """Print a human-readable diagnostic report to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  DIAGNOSTICS – {result.galaxy}")
    print(f"{'=' * 60}")
    print(f"  Data points       : {result.n_points}")
    print(f"  Non-finite values : {result.n_missing}")
    print(f"  Duplicate radii   : {result.n_duplicate_radii}")
    print(f"  Radial range      : {result.r_min_kpc:.2f} – {result.r_max_kpc:.2f} kpc")
    print(f"  Coverage          : {result.r_coverage_kpc:.2f} kpc")
    print(f"  Median spacing    : {result.median_spacing_kpc:.3f} kpc  (CV={result.spacing_cv:.2f})")
    print(f"  Median velocity   : {result.v_median_kms:.1f} km/s  (MAD={result.v_mad_kms:.1f})")
    print(f"  Negative v        : {result.n_negative_v}")
    print(f"  Velocity outliers : {result.n_velocity_outliers}")
    print(f"  Median error      : {result.median_err_kms:.2f} km/s")
    print(f"  Zero/neg errors   : {result.n_zero_errors} / {result.n_negative_errors}")
    print(f"  Heteroscedastic   : {'YES' if result.heteroscedastic else 'no'}")
    print(f"  Median SNR        : {result.median_snr:.1f}")
    print(f"  Low-SNR points    : {result.n_low_snr}")
    slope_flag = "YES" if result.outer_slope_significant else "no"
    print(f"  Outer slope       : {result.outer_slope_kms_per_kpc:.3f} km/s/kpc  (significant={slope_flag})")
    print(f"  Inner slope       : {result.inner_slope_kms_per_kpc:.3f} km/s/kpc")

    status = "PASS" if result.passed else "FAIL"
    print(f"\n  Status: {status}")

    if result.errors:
        print("\n  ERRORS:")
        for err in result.errors:
            print(f"    [ERROR] {err}")
    if result.warnings:
        print("\n  WARNINGS:")
        for warn in result.warnings:
            print(f"    [WARN ] {warn}")
    print()


def export_diagnostics_csv(
    results: List[DiagnosticResult],
    output_dir: str,
) -> str:
    """Write all diagnostic results to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "diagnostics_summary.csv")

    exclude = {"warnings", "errors"}
    fields = [f for f in asdict(results[0]).keys() if f not in exclude]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(fields) + "\n")
        for res in results:
            d = asdict(res)
            row = [str(d[f]) for f in fields]
            fh.write(",".join(row) + "\n")

    return path


def save_diagnostic_report(results: List[DiagnosticResult], output_dir: str) -> str:
    """Write a plain-text report for all galaxies."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "diagnostics_report.txt")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("SCM FRAMEWORK – Motor de Velos\n")
        fh.write("Environment Diagnostics Report\n")
        fh.write("=" * 60 + "\n\n")
        for res in results:
            fh.write(f"Galaxy: {res.galaxy}\n")
            fh.write(f"  Status       : {'PASS' if res.passed else 'FAIL'}\n")
            fh.write(f"  N points     : {res.n_points}\n")
            fh.write(f"  r range [kpc]: {res.r_min_kpc:.2f} – {res.r_max_kpc:.2f}\n")
            fh.write(f"  Median v     : {res.v_median_kms:.1f} km/s\n")
            fh.write(f"  Median SNR   : {res.median_snr:.1f}\n")
            fh.write(f"  Outer slope  : {res.outer_slope_kms_per_kpc:.3f} km/s/kpc\n")
            if res.errors:
                fh.write("  Errors:\n")
                for e in res.errors:
                    fh.write(f"    - {e}\n")
            if res.warnings:
                fh.write("  Warnings:\n")
                for w in res.warnings:
                    fh.write(f"    - {w}\n")
            fh.write("\n")

    return path


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_diagnostic_overview(
    galaxy: str,
    r: np.ndarray,
    v: np.ndarray,
    v_err: np.ndarray,
    result: DiagnosticResult,
    output_dir: str,
) -> None:
    """Save a four-panel diagnostic overview figure."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Diagnostic Overview – {galaxy}", fontsize=13)

    mask = np.isfinite(r) & np.isfinite(v) & np.isfinite(v_err)
    r_f, v_f, ve_f = r[mask], v[mask], v_err[mask]

    # Panel 1: Rotation curve + error bars
    ax = axes[0, 0]
    ax.errorbar(r_f, v_f, yerr=ve_f, fmt="o", color="steelblue", markersize=4, alpha=0.8)
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel("v [km/s]")
    ax.set_title("Rotation curve")
    ax.grid(True, alpha=0.3)

    # Panel 2: SNR profile
    ax = axes[0, 1]
    snr = np.abs(v_f) / ve_f if np.any(ve_f > 0) else np.zeros_like(v_f)
    ax.plot(r_f, snr, "o-", color="darkorange", markersize=4)
    ax.axhline(3.0, color="red", linestyle="--", linewidth=0.9, label="SNR = 3")
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel("SNR")
    ax.set_title(f"Signal-to-noise (median={result.median_snr:.1f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Radial spacing
    ax = axes[1, 0]
    r_sorted = np.sort(r_f)
    spacing = np.diff(r_sorted)
    ax.bar(r_sorted[:-1], spacing, width=spacing * 0.8, color="mediumseagreen", alpha=0.8)
    ax.axhline(result.median_spacing_kpc, color="navy", linestyle="--",
               linewidth=0.9, label=f"Median={result.median_spacing_kpc:.2f} kpc")
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel("Δr [kpc]")
    ax.set_title("Radial spacing")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Error distribution
    ax = axes[1, 1]
    if len(ve_f) > 0:
        ax.hist(ve_f, bins=min(20, len(ve_f)), color="orchid", edgecolor="white", alpha=0.85)
        ax.axvline(result.median_err_kms, color="darkred", linestyle="--",
                   label=f"Median={result.median_err_kms:.2f} km/s")
        ax.set_xlabel("Error [km/s]")
        ax.set_ylabel("Count")
        ax.set_title("Error distribution")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Annotate overall status
    status_color = "green" if result.passed else "red"
    status_text = "PASS" if result.passed else f"FAIL ({len(result.errors)} error(s))"
    fig.text(0.01, 0.01, f"Status: {status_text}",
             fontsize=10, color=status_color, va="bottom")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = os.path.join(output_dir, f"{galaxy}_diagnostics.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_single_galaxy(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a single-galaxy CSV (radius_kpc, velocity_kms[, velocity_err_kms])."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    r = data[:, 0]
    v = data[:, 1]
    v_err = data[:, 2] if data.shape[1] >= 3 else np.ones_like(v) * 5.0
    return r, v, v_err


def load_catalog(
    path: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load a multi-galaxy catalog CSV.

    Columns: galaxy, radius_kpc, velocity_kms, velocity_err_kms
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Catalog not found: {path}")

    catalog: Dict[str, List] = {}
    with open(path, encoding="utf-8") as fh:
        header = next(fh).strip().split(",")
        idx = {col.strip(): i for i, col in enumerate(header)}
        for line in fh:
            parts = line.strip().split(",")
            if not parts or parts[0].startswith("#"):
                continue
            gal = parts[idx["galaxy"]].strip()
            r_val = float(parts[idx["radius_kpc"]])
            v_val = float(parts[idx["velocity_kms"]])
            ve_val = (float(parts[idx["velocity_err_kms"]])
                      if "velocity_err_kms" in idx else 5.0)
            catalog.setdefault(gal, [[], [], []])
            catalog[gal][0].append(r_val)
            catalog[gal][1].append(v_val)
            catalog[gal][2].append(ve_val)

    return {g: (np.array(d[0]), np.array(d[1]), np.array(d[2]))
            for g, d in catalog.items()}


def generate_synthetic_sample(n: int = 4, seed: int = 3) -> Dict[str, Tuple]:
    """Generate synthetic rotation curves for diagnostics testing."""
    rng = np.random.default_rng(seed)
    sample: Dict[str, Tuple] = {}
    for i in range(n):
        name = f"TestGalaxy_{i + 1:03d}"
        n_pts = rng.integers(15, 50)
        r = np.sort(rng.uniform(0.5, 30.0, n_pts))
        v_flat = rng.uniform(150, 250)
        r_c = rng.uniform(1.5, 5.0)
        v_true = v_flat * np.sqrt(1.0 - (r_c / r) * np.arctan(r / r_c))
        noise = rng.uniform(5, 15)
        v_obs = v_true + rng.normal(0, noise, n_pts)
        v_err = noise * rng.uniform(0.7, 1.3, n_pts)
        sample[name] = (r, v_obs, v_err)
    return sample


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SCM Framework – rotation curve environment diagnostics"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input", "-i", default=None,
                       help="Single-galaxy CSV (radius_kpc, velocity_kms[, velocity_err_kms]).")
    group.add_argument("--catalog", "-c", default=None,
                       help="Multi-galaxy catalog CSV (galaxy, radius_kpc, velocity_kms, velocity_err_kms).")
    parser.add_argument("--output", "-o", default="diagnostics",
                        help="Output directory (default: diagnostics/).")
    parser.add_argument("--galaxy", "-g", default="Galaxy",
                        help="Galaxy name, used when --input is provided.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating diagnostic plots.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    os.makedirs(args.output, exist_ok=True)
    make_plots = not args.no_plots

    if args.input:
        try:
            r, v, v_err = load_single_galaxy(args.input)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        sample = {args.galaxy: (r, v, v_err)}
    elif args.catalog:
        try:
            sample = load_catalog(args.catalog)
        except (FileNotFoundError, KeyError) as exc:
            print(f"ERROR loading catalog: {exc}", file=sys.stderr)
            return 1
    else:
        print("No input provided – using synthetic sample for demonstration.")
        sample = generate_synthetic_sample()

    results: List[DiagnosticResult] = []
    for galaxy, (r, v, v_err) in sample.items():
        res = run_diagnostics(galaxy, r, v, v_err)
        print_diagnostic_report(res)
        results.append(res)

        if make_plots:
            plot_diagnostic_overview(galaxy, r, v, v_err, res, args.output)

    # Save CSV and text report
    csv_path = export_diagnostics_csv(results, args.output)
    txt_path = save_diagnostic_report(results, args.output)
    print(f"CSV summary  : {csv_path}")
    print(f"Text report  : {txt_path}")

    # Return non-zero exit code if any galaxy failed
    any_failed = any(not r.passed for r in results)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
