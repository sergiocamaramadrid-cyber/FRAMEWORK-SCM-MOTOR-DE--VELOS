"""
scm_framework_model_comparison.py
===================================
Framework SCM – Motor de Velos
Linear model comparison for galaxy rotation-curve environmental analysis.

This module fits and compares four nested OLS regression models that relate
the SCM velocity factor F3 to stellar mass and environmental/geometric
covariates on the same galaxy catalog, enabling principled model selection.

Models compared
---------------
  M0 : F3 ~ logM
  M1 : F3 ~ logM + env_proxy
  M2 : F3 ~ logM + env_proxy + inc_deg
  M3 : F3 ~ logM + env_proxy + inc_deg + env_proxy:inc_deg

Metrics reported per model
--------------------------
  R²               coefficient of determination
  R² adjusted      penalised for number of predictors
  AIC              Akaike Information Criterion
  BIC              Bayesian Information Criterion
  ΔAIC             AIC relative to best model
  ΔBIC             BIC relative to best model
  p-values         per-coefficient significance
  N                number of galaxies used after cleaning

Outputs
-------
  results/scm_model_comparison/model_comparison.json
  results/scm_model_comparison/model_coefficients.csv
  results/scm_model_comparison/model_selection_summary.txt

Usage
-----
    python scripts/scm_framework_model_comparison.py \\
        --catalog data/catalog.csv \\
        --output  results/scm_model_comparison/

    Required catalog columns:
        f3          – SCM velocity factor (dependent variable)
        log_mass    – log10 stellar / total mass [log(M☉)]
        env_proxy   – environmental proxy (e.g. local density, isolation index)
        inc_deg     – disk inclination [degrees]

    Optional catalog column:
        galaxy      – galaxy identifier (used in summary only)

    All other columns are ignored.
"""

import argparse
import json
import math
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy dependencies – fail gracefully so the module can be imported
# even in stripped environments.
# ---------------------------------------------------------------------------

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    warnings.warn("pandas not available – catalog loading requires pandas.")

try:
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False
    warnings.warn(
        "statsmodels not available – falling back to numpy-based OLS. "
        "p-values and log-likelihood based AIC/BIC will be approximate."
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAMES = ["M0", "M1", "M2", "M3"]
MODEL_FORMULAS = {
    "M0": "F3 ~ logM",
    "M1": "F3 ~ logM + env_proxy",
    "M2": "F3 ~ logM + env_proxy + inc_deg",
    "M3": "F3 ~ logM + env_proxy + inc_deg + env_proxy:inc_deg",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_catalog(path: str) -> "pd.DataFrame":
    """Load and validate the galaxy catalog CSV."""
    if not HAS_PANDAS:
        raise RuntimeError("pandas is required to load the catalog.")

    df = pd.read_csv(path)

    required = {"f3", "log_mass", "env_proxy", "inc_deg"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Catalog is missing required columns: {sorted(missing_cols)}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Drop rows with any NaN in required columns
    n_before = len(df)
    df = df.dropna(subset=sorted(required)).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        warnings.warn(f"Dropped {n_dropped} rows with missing values in required columns.")

    if len(df) < 4:
        raise ValueError(
            f"Too few usable rows ({len(df)}) after cleaning – need at least 4."
        )

    return df


# ---------------------------------------------------------------------------
# Design matrix helpers
# ---------------------------------------------------------------------------

def build_design_matrix(df: "pd.DataFrame", model_id: str) -> np.ndarray:
    """Return the design matrix X (with intercept column) for *model_id*."""
    intercept = np.ones(len(df))
    log_mass  = df["log_mass"].to_numpy(dtype=float)
    env_proxy = df["env_proxy"].to_numpy(dtype=float)
    inc_deg   = df["inc_deg"].to_numpy(dtype=float)
    interaction = env_proxy * inc_deg

    columns = {
        "M0": [intercept, log_mass],
        "M1": [intercept, log_mass, env_proxy],
        "M2": [intercept, log_mass, env_proxy, inc_deg],
        "M3": [intercept, log_mass, env_proxy, inc_deg, interaction],
    }
    return np.column_stack(columns[model_id])


def coef_names(model_id: str) -> List[str]:
    """Return coefficient labels matching the design matrix columns."""
    names = {
        "M0": ["intercept", "logM"],
        "M1": ["intercept", "logM", "env_proxy"],
        "M2": ["intercept", "logM", "env_proxy", "inc_deg"],
        "M3": ["intercept", "logM", "env_proxy", "inc_deg", "env_proxy:inc_deg"],
    }
    return names[model_id]


# ---------------------------------------------------------------------------
# OLS fitting
# ---------------------------------------------------------------------------

def _fit_numpy(
    y: np.ndarray, X: np.ndarray, model_id: str, n: int
) -> Dict:
    """Minimal OLS via numpy.linalg.lstsq.  AIC/BIC use Gaussian likelihood."""
    coefs, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if len(residuals) == 0:
        ss_res = np.sum((y - X @ coefs) ** 2)
    else:
        ss_res = float(residuals[0])

    ss_tot = np.sum((y - y.mean()) ** 2)
    k = X.shape[1]  # number of parameters (includes intercept)
    dof = n - k

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / dof if dof > 0 else float("nan")

    sigma2 = ss_res / n  # MLE variance estimate
    if sigma2 > 0:
        log_lik = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1.0)
    else:
        log_lik = float("nan")

    aic = 2 * k - 2 * log_lik if not math.isnan(log_lik) else float("nan")
    bic = k * math.log(n) - 2 * log_lik if not math.isnan(log_lik) else float("nan")

    # p-values via t-distribution (approximate when statsmodels unavailable)
    try:
        from scipy import stats as scipy_stats
        XtX_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(XtX_inv) * (ss_res / dof), 0.0))
        t_vals = coefs / np.where(se > 0, se, np.nan)
        pvals = [
            float(2 * scipy_stats.t.sf(abs(t), df=dof)) if not math.isnan(t) else float("nan")
            for t in t_vals
        ]
    except ImportError:
        pvals = [float("nan")] * len(coefs)

    names = coef_names(model_id)
    coef_dict = {
        nm: {"coef": float(c), "pvalue": float(p)}
        for nm, c, p in zip(names, coefs, pvals)
    }

    return {
        "model": model_id,
        "formula": MODEL_FORMULAS[model_id],
        "n": n,
        "r2": r2,
        "r2_adj": r2_adj,
        "aic": aic,
        "bic": bic,
        "log_likelihood": log_lik,
        "k": k,
        "coefficients": coef_dict,
    }


def _fit_statsmodels(
    y: np.ndarray, X: np.ndarray, model_id: str, n: int
) -> Dict:
    """OLS via statsmodels for exact AIC/BIC and t-test p-values."""
    result = sm.OLS(y, X).fit()

    names = coef_names(model_id)
    coef_dict = {
        nm: {
            "coef": float(result.params[i]),
            "std_err": float(result.bse[i]),
            "t_stat": float(result.tvalues[i]),
            "pvalue": float(result.pvalues[i]),
        }
        for i, nm in enumerate(names)
    }

    return {
        "model": model_id,
        "formula": MODEL_FORMULAS[model_id],
        "n": n,
        "r2": float(result.rsquared),
        "r2_adj": float(result.rsquared_adj),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "log_likelihood": float(result.llf),
        "k": int(result.df_model) + 1,  # predictors + intercept
        "coefficients": coef_dict,
    }


def fit_model(df: "pd.DataFrame", model_id: str) -> Dict:
    """Fit a single model and return its result dictionary."""
    y = df["f3"].to_numpy(dtype=float)
    X = build_design_matrix(df, model_id)
    n = len(df)

    if HAS_SM:
        return _fit_statsmodels(y, X, model_id, n)
    return _fit_numpy(y, X, model_id, n)


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(df: "pd.DataFrame") -> List[Dict]:
    """Fit all four models and annotate with ΔAIC / ΔBIC."""
    results = [fit_model(df, mid) for mid in MODEL_NAMES]

    aic_vals = [r["aic"] for r in results]
    bic_vals = [r["bic"] for r in results]

    aic_min = min(v for v in aic_vals if not math.isnan(v))
    bic_min = min(v for v in bic_vals if not math.isnan(v))

    for r in results:
        r["delta_aic"] = r["aic"] - aic_min if not math.isnan(r["aic"]) else float("nan")
        r["delta_bic"] = r["bic"] - bic_min if not math.isnan(r["bic"]) else float("nan")

    return results


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_json(results: List[Dict], path: str) -> None:
    """Serialise full results to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, allow_nan=True)


def write_coefficients_csv(results: List[Dict], path: str) -> None:
    """Write one row per (model, coefficient) with all available stats."""
    if not HAS_PANDAS:
        # Fallback: write manually
        rows = []
        for r in results:
            for cname, cstats in r["coefficients"].items():
                row = {"model": r["model"], "term": cname}
                row.update(cstats)
                rows.append(row)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if rows:
            header = list(rows[0].keys())
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(",".join(header) + "\n")
                for row in rows:
                    fh.write(",".join(str(row.get(h, "")) for h in header) + "\n")
        return

    rows = []
    for r in results:
        for cname, cstats in r["coefficients"].items():
            row = {"model": r["model"], "term": cname}
            row.update(cstats)
            rows.append(row)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6g")


def write_summary_txt(results: List[Dict], path: str, catalog_path: str) -> None:
    """Write the human-readable model selection summary."""
    best_aic = min(results, key=lambda r: r["aic"] if not math.isnan(r["aic"]) else float("inf"))
    best_bic = min(results, key=lambda r: r["bic"] if not math.isnan(r["bic"]) else float("inf"))

    lines = [
        "=" * 70,
        "  SCM Framework – Model Comparison Summary",
        "=" * 70,
        f"  Catalog  : {catalog_path}",
        f"  N (used) : {results[0]['n']}",
        f"  Backend  : {'statsmodels OLS' if HAS_SM else 'numpy lstsq (approximate)'}",
        "",
        "  Dependent variable : F3",
        "  Predictors tested  :",
        "    logM        – log10 stellar/total mass",
        "    env_proxy   – environmental proxy",
        "    inc_deg     – disk inclination [°]",
        "    env_proxy:inc_deg – interaction term",
        "",
        "-" * 70,
        f"  {'Model':<6} {'Formula':<45} {'N':>5}",
        "-" * 70,
    ]
    for r in results:
        lines.append(f"  {r['model']:<6} {r['formula']:<45} {r['n']:>5}")

    lines += [
        "",
        "-" * 70,
        f"  {'Model':<6} {'R²':>8} {'R²adj':>8} {'AIC':>10} {'BIC':>10} "
        f"{'ΔAIC':>8} {'ΔBIC':>8}",
        "-" * 70,
    ]

    def _fmt(v: float, width: int = 8, decimals: int = 4) -> str:
        if math.isnan(v):
            return " " * (width - 3) + "nan"
        return f"{v:{width}.{decimals}f}"

    for r in results:
        marker = ""
        if r["model"] == best_aic["model"] and r["model"] == best_bic["model"]:
            marker = " ← best AIC & BIC"
        elif r["model"] == best_aic["model"]:
            marker = " ← best AIC"
        elif r["model"] == best_bic["model"]:
            marker = " ← best BIC"
        lines.append(
            f"  {r['model']:<6}"
            f"{_fmt(r['r2']):>8}"
            f"{_fmt(r['r2_adj']):>8}"
            f"{_fmt(r['aic'], 10, 2):>10}"
            f"{_fmt(r['bic'], 10, 2):>10}"
            f"{_fmt(r['delta_aic'], 8, 2):>8}"
            f"{_fmt(r['delta_bic'], 8, 2):>8}"
            f"{marker}"
        )

    lines += [
        "",
        "  Rule of thumb (ΔAIC / ΔBIC):",
        "    < 2  → substantial support for the model",
        "    2–7  → considerably less support",
        "    > 10 → essentially no support",
        "",
        "-" * 70,
        "  Coefficient p-values",
        "-" * 70,
    ]

    for r in results:
        lines.append(f"\n  [{r['model']}]  {r['formula']}")
        lines.append(f"  {'Term':<30} {'Coef':>12} {'p-value':>10}")
        lines.append(f"  {'-'*30} {'-'*12} {'-'*10}")
        for term, stats in r["coefficients"].items():
            coef_val = stats.get("coef", float("nan"))
            pval     = stats.get("pvalue", float("nan"))
            sig      = ""
            if not math.isnan(pval):
                if pval < 0.001:
                    sig = "***"
                elif pval < 0.01:
                    sig = "**"
                elif pval < 0.05:
                    sig = "*"
                elif pval < 0.1:
                    sig = "."
            lines.append(
                f"  {term:<30} {coef_val:>12.4f} {pval:>10.4f}  {sig}"
            )

    lines += [
        "",
        "  Significance codes: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1",
        "",
        "=" * 70,
        f"  Best model by AIC : {best_aic['model']} ({best_aic['formula']})",
        f"  Best model by BIC : {best_bic['model']} ({best_bic['formula']})",
        "=" * 70,
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SCM Framework – compare linear models for F3 vs. stellar mass, "
            "environment, and inclination."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--catalog",
        required=True,
        metavar="PATH",
        help=(
            "CSV catalog with columns: f3, log_mass, env_proxy, inc_deg "
            "(optionally: galaxy)."
        ),
    )
    parser.add_argument(
        "--output",
        default="results/scm_model_comparison",
        metavar="DIR",
        help="Output directory (default: results/scm_model_comparison).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not HAS_PANDAS:
        print(
            "ERROR: pandas is required. Install with:  pip install pandas",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"[SCM] Loading catalog: {args.catalog}")
    try:
        df = load_catalog(args.catalog)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"[SCM] Usable galaxies: {len(df)}")

    # ------------------------------------------------------------------
    # Fit models
    # ------------------------------------------------------------------
    print("[SCM] Fitting models M0 – M3 …")
    results = compare_models(df)

    for r in results:
        print(
            f"  {r['model']}  R²={r['r2']:.4f}  R²adj={r['r2_adj']:.4f}  "
            f"AIC={r['aic']:.2f}  BIC={r['bic']:.2f}  "
            f"ΔAIC={r['delta_aic']:.2f}  ΔBIC={r['delta_bic']:.2f}"
        )

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    out = args.output
    json_path  = os.path.join(out, "model_comparison.json")
    csv_path   = os.path.join(out, "model_coefficients.csv")
    txt_path   = os.path.join(out, "model_selection_summary.txt")

    write_json(results, json_path)
    print(f"[SCM] Written: {json_path}")

    write_coefficients_csv(results, csv_path)
    print(f"[SCM] Written: {csv_path}")

    write_summary_txt(results, txt_path, args.catalog)
    print(f"[SCM] Written: {txt_path}")

    # ------------------------------------------------------------------
    # Print summary to stdout
    # ------------------------------------------------------------------
    best_aic = min(results, key=lambda r: r["aic"] if not math.isnan(r["aic"]) else float("inf"))
    best_bic = min(results, key=lambda r: r["bic"] if not math.isnan(r["bic"]) else float("inf"))
    print(f"\n[SCM] Best model by AIC: {best_aic['model']} ({best_aic['formula']})")
    print(f"[SCM] Best model by BIC: {best_bic['model']} ({best_bic['formula']})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
