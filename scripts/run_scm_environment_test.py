#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SCM Environment Test (Phase A/B)

Cruza el catálogo F3 (slope_tail, delta_f3) con un proxy ambiental
y evalúa:

- Correlación Spearman
- Regresión OLS (HC3 robust)
- Comparación AIC (modelo con vs sin entorno)

Outputs:
- results/scm_environment/summary.json
- results/scm_environment/merged_catalog.csv
"""

import os
import json
import argparse
import pandas as pd
import numpy as np

from scipy.stats import spearmanr
import statsmodels.api as sm


# =========================================================
# Utils
# =========================================================

def load_and_clean(f3_path, env_path):
    f3 = pd.read_csv(f3_path)
    env = pd.read_csv(env_path)

    # Normalización de nombres
    if "galaxy" not in f3.columns:
        raise ValueError("F3 catalog must contain 'galaxy' column")

    if "galaxy" not in env.columns:
        # intento flexible
        for c in env.columns:
            if "gal" in c.lower():
                env = env.rename(columns={c: "galaxy"})
                break

    if "env_proxy" not in env.columns:
        raise ValueError("Env catalog must contain 'env_proxy' column")

    # merge
    df = pd.merge(f3, env, on="galaxy", how="inner")

    # filtros SCM estándar
    if "outer_fit_ok" in df.columns:
        df = df[df["outer_fit_ok"] == True]

    if "tail_n" in df.columns:
        df = df[df["tail_n"] >= 4]

    df = df.dropna(subset=["slope_tail", "env_proxy"])

    return df


def compute_spearman(df):
    rho, p = spearmanr(df["env_proxy"], df["slope_tail"])
    return float(rho), float(p)


def run_regression(df):
    # Modelo base (solo intercepto)
    X0 = np.ones(len(df))
    model0 = sm.OLS(df["slope_tail"], X0).fit()

    # Modelo con entorno
    X1 = sm.add_constant(df["env_proxy"])
    model1 = sm.OLS(df["slope_tail"], X1).fit(cov_type="HC3")

    return model0, model1


def summarize(df, rho, p, m0, m1):
    return {
        "N": int(len(df)),
        "spearman": {
            "rho": rho,
            "p_value": p
        },
        "regression": {
            "beta_env": float(m1.params.iloc[1]),
            "p_env": float(m1.pvalues.iloc[1]),
            "AIC_base": float(m0.aic),
            "AIC_env": float(m1.aic),
            "delta_AIC": float(m0.aic - m1.aic)
        }
    }


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="SCM Environment Test")
    parser.add_argument("--f3", required=True, help="F3 catalog CSV")
    parser.add_argument("--env", required=True, help="Environment CSV")
    parser.add_argument("--outdir", default="results/scm_environment")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load
    df = load_and_clean(args.f3, args.env)

    if len(df) < 10:
        raise ValueError("Sample too small after cleaning (<10 galaxies)")

    # Stats
    rho, p = compute_spearman(df)
    m0, m1 = run_regression(df)

    summary = summarize(df, rho, p, m0, m1)

    # Save outputs
    df.to_csv(os.path.join(args.outdir, "merged_catalog.csv"), index=False)

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # Print
    print("\n===== SCM ENVIRONMENT TEST =====")
    print(f"N = {summary['N']}")
    print(f"Spearman rho = {summary['spearman']['rho']:.3f} (p = {summary['spearman']['p_value']:.3e})")
    print(f"beta_env = {summary['regression']['beta_env']:.3f} (p = {summary['regression']['p_env']:.3e})")
    print(f"ΔAIC = {summary['regression']['delta_AIC']:.2f}")
    print("================================\n")


if __name__ == "__main__":
    main()
