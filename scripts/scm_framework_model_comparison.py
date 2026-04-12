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
  M3 : F3 ~ logM + env_proxy + inc_deg + env_proxy_c:inc_c  (centered interaction)

Design decisions
----------------
  * All four models are fitted on the **exact same subsample**: rows with no
    NaN in any of the four required columns (as needed by M3).  This makes
    AIC, BIC, ΔAIC, and ΔBIC directly comparable across models.
  * p-values use HC3 heteroskedasticity-robust standard errors.
    AIC and BIC are computed from the standard (non-robust) log-likelihood,
    which is the only consistent definition for Gaussian OLS.
  * The interaction term in M3 uses mean-centred predictors
    (env_proxy_c = env_proxy − mean, inc_c = inc_deg − mean) to reduce
    multicollinearity and improve interpretability of the main-effect
    coefficients.
  * statsmodels is a hard dependency.  The script fails with a clear message
    if it is not installed rather than silently degrading to numpy estimates.

Metrics reported per model
--------------------------
  R²               coefficient of determination
  R² adjusted      penalised for number of predictors
  AIC              Akaike Information Criterion  (standard OLS log-likelihood)
  BIC              Bayesian Information Criterion
  ΔAIC             AIC relative to best model
  ΔBIC             BIC relative to best model
  ΔAIC vs M0       improvement relative to baryonic baseline
  ΔBIC vs M0       improvement relative to baryonic baseline
  p-values         per-coefficient (HC3 robust)
  N                number of galaxies used (identical for all models)

Selection flags (in JSON and summary)
--------------------------------------
  best_aic_model           model ID with lowest AIC
  best_bic_model           model ID with lowest BIC
  environment_term_supported   True if ΔAIC(M1 vs M0) ≤ −2
  interaction_term_supported   True if ΔAIC(M3 vs M2) ≤ −2

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

    The catalog must contain one column for each role below.  Accepted
    aliases (first match wins, case-sensitive):

        F3 target   : F3_SCM, slope_tail, F3, f3
        Stellar mass: logM, logMbar, log_mass
        Environment : env_proxy
        Inclination : inc_deg, Inc

    Optional column:
        galaxy      – galaxy identifier (used in summary only)

    All other columns are ignored.

Requirements
------------
    pip install pandas statsmodels numpy
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Hard dependencies – fail early with actionable messages
# ---------------------------------------------------------------------------

try:
    import pandas as pd
except ImportError:
    print(
        "ERROR: pandas is required.  Install with:  pip install pandas",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import statsmodels.api as sm
except ImportError:
    print(
        "ERROR: statsmodels is required.  Install with:  pip install statsmodels\n"
        "       Running without statsmodels would produce inconsistent AIC/BIC values\n"
        "       and non-robust p-values.  Please install it before proceeding.",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Column alias resolution
# ---------------------------------------------------------------------------

# Ordered lists: first alias found in the catalog wins.
_ALIASES: Dict[str, List[str]] = {
    "f3":        ["F3_SCM", "slope_tail", "F3", "f3"],
    "log_mass":  ["logM", "logMbar", "log_mass"],
    "env_proxy": ["env_proxy"],
    "inc_deg":   ["inc_deg", "Inc"],
}


def _resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Return a mapping {internal_name: actual_column_name} for the catalog."""
    resolved: Dict[str, str] = {}
    missing: List[str] = []
    for internal, aliases in _ALIASES.items():
        found = next((a for a in aliases if a in df.columns), None)
        if found is None:
            missing.append(f"{internal!r}  (tried: {aliases})")
        else:
            resolved[internal] = found
    if missing:
        raise ValueError(
            "Catalog is missing columns for the following roles:\n"
            + "\n".join(f"  {m}" for m in missing)
            + f"\n\nFound columns: {list(df.columns)}"
        )
    return resolved


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAMES = ["M0", "M1", "M2", "M3"]
MODEL_FORMULAS = {
    "M0": "F3 ~ logM",
    "M1": "F3 ~ logM + env_proxy",
    "M2": "F3 ~ logM + env_proxy + inc_deg",
    "M3": "F3 ~ logM + env_proxy + inc_deg + env_proxy_c:inc_c",
}

# Threshold for "appreciable improvement" in ΔAIC / ΔBIC
_SUPPORT_THRESHOLD = -2.0

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_catalog(path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load, resolve columns, and clean the galaxy catalog CSV.

    Returns
    -------
    df : pd.DataFrame
        Cleaned catalog with internal canonical column names.
    col_map : dict
        Mapping from internal name to original column name (for reporting).
    """
    df_raw = pd.read_csv(path)
    col_map = _resolve_columns(df_raw)

    # Rename to canonical internal names so the rest of the code is uniform
    rename = {v: k for k, v in col_map.items() if v != k}
    df = df_raw.rename(columns=rename)

    # Drop rows with any NaN in the four required columns (same set for M0–M3,
    # so that AIC/BIC are comparable across all models).
    required = list(_ALIASES.keys())
    n_before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(
            f"[SCM] WARNING: Dropped {n_dropped} row(s) with missing values "
            f"in required columns.  Same {len(df)}-galaxy subsample used for M0–M3.",
            file=sys.stderr,
        )

    if len(df) < 5:
        raise ValueError(
            f"Too few usable rows ({len(df)}) after cleaning – need at least 5."
        )

    return df, col_map


# ---------------------------------------------------------------------------
# Design matrix helpers
# ---------------------------------------------------------------------------


def build_design_matrix(df: pd.DataFrame, model_id: str) -> np.ndarray:
    """Return the design matrix X (with intercept) for *model_id*.

    For M3 the interaction term uses mean-centred env_proxy and inc_deg to
    reduce multicollinearity and improve coefficient interpretability.
    """
    intercept = np.ones(len(df))
    log_mass  = df["log_mass"].to_numpy(dtype=float)
    env_proxy = df["env_proxy"].to_numpy(dtype=float)
    inc_deg   = df["inc_deg"].to_numpy(dtype=float)

    # Centred versions (used only by M3)
    env_c = env_proxy - env_proxy.mean()
    inc_c = inc_deg   - inc_deg.mean()
    interaction = env_c * inc_c

    columns: Dict[str, List[np.ndarray]] = {
        "M0": [intercept, log_mass],
        "M1": [intercept, log_mass, env_proxy],
        "M2": [intercept, log_mass, env_proxy, inc_deg],
        "M3": [intercept, log_mass, env_proxy, inc_deg, interaction],
    }
    return np.column_stack(columns[model_id])


def coef_names(model_id: str) -> List[str]:
    """Return coefficient labels matching the design matrix columns."""
    names: Dict[str, List[str]] = {
        "M0": ["intercept", "logM"],
        "M1": ["intercept", "logM", "env_proxy"],
        "M2": ["intercept", "logM", "env_proxy", "inc_deg"],
        "M3": ["intercept", "logM", "env_proxy", "inc_deg", "env_proxy_c:inc_c"],
    }
    return names[model_id]


# ---------------------------------------------------------------------------
# OLS fitting (statsmodels, HC3 robust p-values)
# ---------------------------------------------------------------------------


def fit_model(df: pd.DataFrame, model_id: str) -> Dict:
    """Fit a single OLS model and return its result dictionary.

    AIC and BIC are derived from the standard (homoskedastic) log-likelihood
    so they remain comparable across models.  p-values use HC3 robust
    standard errors.
    """
    y = df["f3"].to_numpy(dtype=float)
    X = build_design_matrix(df, model_id)
    n = len(df)

    ols_base = sm.OLS(y, X)

    # Standard fit – used for AIC, BIC, R², log-likelihood
    res_ols = ols_base.fit()
    # HC3 robust fit – used for standard errors and p-values
    res_hc3 = ols_base.fit(cov_type="HC3")

    names = coef_names(model_id)
    coef_dict = {
        nm: {
            "coef":    float(res_ols.params[i]),
            "std_err": float(res_hc3.bse[i]),
            "t_stat":  float(res_hc3.tvalues[i]),
            "pvalue":  float(res_hc3.pvalues[i]),
        }
        for i, nm in enumerate(names)
    }

    return {
        "model":          model_id,
        "formula":        MODEL_FORMULAS[model_id],
        "n":              n,
        "r2":             float(res_ols.rsquared),
        "r2_adj":         float(res_ols.rsquared_adj),
        "aic":            float(res_ols.aic),
        "bic":            float(res_ols.bic),
        "log_likelihood": float(res_ols.llf),
        "k":              int(res_ols.df_model) + 1,
        "coefficients":   coef_dict,
    }


# ---------------------------------------------------------------------------
# Model comparison and selection flags
# ---------------------------------------------------------------------------


def compare_models(df: pd.DataFrame) -> Tuple[List[Dict], Dict]:
    """Fit M0–M3 on the same subsample and compute comparison metrics.

    Returns
    -------
    results : list of dicts
        Per-model statistics with delta_aic / delta_bic annotated.
    selection : dict
        High-level selection summary (best models, support flags, deltas vs M0).
    """
    results = [fit_model(df, mid) for mid in MODEL_NAMES]

    by_id = {r["model"]: r for r in results}

    aic_min = min(r["aic"] for r in results)
    bic_min = min(r["bic"] for r in results)
    aic_m0  = by_id["M0"]["aic"]
    bic_m0  = by_id["M0"]["bic"]

    for r in results:
        r["delta_aic"]      = r["aic"] - aic_min
        r["delta_bic"]      = r["bic"] - bic_min
        r["delta_aic_vs_m0"] = r["aic"] - aic_m0
        r["delta_bic_vs_m0"] = r["bic"] - bic_m0

    best_aic = min(results, key=lambda r: r["aic"])
    best_bic = min(results, key=lambda r: r["bic"])

    # Support flags: ΔAIC(Mx vs baseline) ≤ −2  → appreciable improvement
    env_supported = (by_id["M1"]["aic"] - aic_m0) <= _SUPPORT_THRESHOLD
    int_supported = (by_id["M3"]["aic"] - by_id["M2"]["aic"]) <= _SUPPORT_THRESHOLD

    selection = {
        "best_aic_model":              best_aic["model"],
        "best_bic_model":              best_bic["model"],
        "environment_term_supported":  env_supported,
        "interaction_term_supported":  int_supported,
        "delta_aic_m1_vs_m0":         by_id["M1"]["delta_aic_vs_m0"],
        "delta_bic_m1_vs_m0":         by_id["M1"]["delta_bic_vs_m0"],
        "delta_aic_m3_vs_m2":         by_id["M3"]["aic"] - by_id["M2"]["aic"],
        "delta_bic_m3_vs_m2":         by_id["M3"]["bic"] - by_id["M2"]["bic"],
        "support_threshold_aic":      _SUPPORT_THRESHOLD,
        "note_pvalues":               "HC3 robust standard errors",
        "note_aic_bic":               "standard OLS log-likelihood (comparable across M0–M3)",
        "note_interaction":           "M3 interaction uses mean-centred env_proxy and inc_deg",
        "note_subsample":             "identical N for all models (dropna on all four columns)",
    }

    return results, selection


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_json(results: List[Dict], selection: Dict, path: str) -> None:
    """Serialise full results + selection summary to JSON."""
    payload = {"selection_summary": selection, "models": results}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, allow_nan=True)


def write_coefficients_csv(results: List[Dict], path: str) -> None:
    """Write one row per (model, coefficient) with all available stats."""
    rows = []
    for r in results:
        for cname, cstats in r["coefficients"].items():
            row = {"model": r["model"], "term": cname}
            row.update(cstats)
            rows.append(row)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6g")


def write_summary_txt(
    results: List[Dict],
    selection: Dict,
    col_map: Dict[str, str],
    path: str,
    catalog_path: str,
) -> None:
    """Write the human-readable model selection summary."""
    best_aic_id = selection["best_aic_model"]
    best_bic_id = selection["best_bic_model"]

    def _fmt(v: float, width: int = 8, decimals: int = 4) -> str:
        if math.isnan(v):
            return " " * (width - 3) + "nan"
        return f"{v:{width}.{decimals}f}"

    def _yn(flag: bool) -> str:
        return "yes" if flag else "no"

    lines = [
        "=" * 72,
        "  SCM Framework – Model Comparison Summary",
        "=" * 72,
        f"  Catalog  : {catalog_path}",
        f"  N (used) : {results[0]['n']}  (identical subsample for M0–M3)",
        f"  Backend  : statsmodels OLS · p-values via HC3 robust std errors",
        "",
        "  Column mapping (catalog → internal):",
    ]
    for internal, actual in col_map.items():
        arrow = f"  {actual}" if actual != internal else ""
        lines.append(f"    {internal:<12} ← {actual}")

    lines += [
        "",
        "  Dependent variable : F3",
        "  Predictors tested  :",
        "    logM              – log10 stellar/total mass",
        "    env_proxy         – environmental proxy",
        "    inc_deg           – disk inclination [°]",
        "    env_proxy_c:inc_c – mean-centred interaction (M3 only)",
        "",
        "-" * 72,
        f"  {'Model':<6} {'Formula':<50} {'N':>5}",
        "-" * 72,
    ]
    for r in results:
        lines.append(f"  {r['model']:<6} {r['formula']:<50} {r['n']:>5}")

    lines += [
        "",
        "-" * 72,
        f"  {'Model':<6} {'R²':>8} {'R²adj':>8} {'AIC':>10} {'BIC':>10} "
        f"{'ΔAIC':>8} {'ΔBIC':>8} {'ΔvM0':>8}",
        "-" * 72,
    ]

    for r in results:
        marker = ""
        if r["model"] == best_aic_id and r["model"] == best_bic_id:
            marker = " ← best AIC & BIC"
        elif r["model"] == best_aic_id:
            marker = " ← best AIC"
        elif r["model"] == best_bic_id:
            marker = " ← best BIC"
        lines.append(
            f"  {r['model']:<6}"
            f"{_fmt(r['r2']):>8}"
            f"{_fmt(r['r2_adj']):>8}"
            f"{_fmt(r['aic'], 10, 2):>10}"
            f"{_fmt(r['bic'], 10, 2):>10}"
            f"{_fmt(r['delta_aic'], 8, 2):>8}"
            f"{_fmt(r['delta_bic'], 8, 2):>8}"
            f"{_fmt(r['delta_aic_vs_m0'], 8, 2):>8}"
            f"{marker}"
        )

    lines += [
        "",
        "  Rule of thumb (ΔAIC / ΔBIC):",
        "    ≤ −6 → strong improvement",
        "    ≤ −2 → appreciable improvement  [threshold used for support flags]",
        "     0   → baseline (M0)",
        "    > 0  → worse than baseline",
        "",
        "-" * 72,
        "  Selection flags",
        "-" * 72,
        f"  best_aic_model             : {best_aic_id}  ({results[MODEL_NAMES.index(best_aic_id)]['formula']})",
        f"  best_bic_model             : {best_bic_id}  ({results[MODEL_NAMES.index(best_bic_id)]['formula']})",
        f"  environment_term_supported : {_yn(selection['environment_term_supported'])}"
        f"  (ΔAIC M1 vs M0 = {selection['delta_aic_m1_vs_m0']:+.2f})",
        f"  interaction_term_supported : {_yn(selection['interaction_term_supported'])}"
        f"  (ΔAIC M3 vs M2 = {selection['delta_aic_m3_vs_m2']:+.2f})",
        "",
        "-" * 72,
        "  Coefficient p-values (HC3 robust)",
        "-" * 72,
    ]

    for r in results:
        lines.append(f"\n  [{r['model']}]  {r['formula']}")
        lines.append(f"  {'Term':<30} {'Coef':>12} {'Std Err':>10} {'p-value':>10}")
        lines.append(f"  {'-'*30} {'-'*12} {'-'*10} {'-'*10}")
        for term, stats in r["coefficients"].items():
            coef_val = stats.get("coef", float("nan"))
            std_err  = stats.get("std_err", float("nan"))
            pval     = stats.get("pvalue", float("nan"))
            sig = ""
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
                f"  {term:<30} {coef_val:>12.4f} {std_err:>10.4f} {pval:>10.4f}  {sig}"
            )

    lines += [
        "",
        "  Significance codes: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1",
        "",
        "=" * 72,
        f"  Best model by AIC : {best_aic_id}  ({results[MODEL_NAMES.index(best_aic_id)]['formula']})",
        f"  Best model by BIC : {best_bic_id}  ({results[MODEL_NAMES.index(best_bic_id)]['formula']})",
        "",
        f"  Environmental support relative to baryonic baseline : "
        f"{_yn(selection['environment_term_supported'])}",
        f"  Interaction support relative to additive model      : "
        f"{_yn(selection['interaction_term_supported'])}",
        "=" * 72,
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
            "CSV catalog.  Accepted column aliases – "
            "F3: F3_SCM/slope_tail/F3/f3; "
            "mass: logM/logMbar/log_mass; "
            "env: env_proxy; "
            "inc: inc_deg/Inc."
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

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"[SCM] Loading catalog: {args.catalog}")
    try:
        df, col_map = load_catalog(args.catalog)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"[SCM] Usable galaxies (same subsample for M0–M3): {len(df)}")
    print(f"[SCM] Column mapping: { {k: v for k, v in col_map.items()} }")

    # ------------------------------------------------------------------
    # Fit models
    # ------------------------------------------------------------------
    print("[SCM] Fitting models M0 – M3 (statsmodels OLS, HC3 p-values) …")
    results, selection = compare_models(df)

    for r in results:
        print(
            f"  {r['model']}  R²={r['r2']:.4f}  R²adj={r['r2_adj']:.4f}  "
            f"AIC={r['aic']:.2f}  BIC={r['bic']:.2f}  "
            f"ΔAIC={r['delta_aic']:.2f}  ΔBIC={r['delta_bic']:.2f}  "
            f"ΔAICvM0={r['delta_aic_vs_m0']:+.2f}"
        )

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    out = args.output
    json_path = os.path.join(out, "model_comparison.json")
    csv_path  = os.path.join(out, "model_coefficients.csv")
    txt_path  = os.path.join(out, "model_selection_summary.txt")

    write_json(results, selection, json_path)
    print(f"[SCM] Written: {json_path}")

    write_coefficients_csv(results, csv_path)
    print(f"[SCM] Written: {csv_path}")

    write_summary_txt(results, selection, col_map, txt_path, args.catalog)
    print(f"[SCM] Written: {txt_path}")

    # ------------------------------------------------------------------
    # Final verdict to stdout
    # ------------------------------------------------------------------
    def _yn(flag: bool) -> str:
        return "yes" if flag else "no"

    print(
        f"\n[SCM] Best model by AIC: {selection['best_aic_model']}  "
        f"Best by BIC: {selection['best_bic_model']}"
    )
    print(
        f"[SCM] Environmental support relative to baryonic baseline : "
        f"{_yn(selection['environment_term_supported'])} "
        f"(ΔAIC M1 vs M0 = {selection['delta_aic_m1_vs_m0']:+.2f})"
    )
    print(
        f"[SCM] Interaction support relative to additive model      : "
        f"{_yn(selection['interaction_term_supported'])} "
        f"(ΔAIC M3 vs M2 = {selection['delta_aic_m3_vs_m2']:+.2f})"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
