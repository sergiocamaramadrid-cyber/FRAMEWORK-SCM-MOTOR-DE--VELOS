"""
analyze_predictions.py – Framework SCM · Motor de Velos

Analiza predicciones de velocidad de rotación galáctica comparando el modelo
BTFR clásico con el modelo interpolado SCM, aplica corrección de offset global
y genera tabla, resumen JSON/TXT y figuras reproducibles.

Uso básico (desde la raíz del repo):
    python scripts/analyze_predictions.py

Uso con rutas explícitas:
    python scripts/analyze_predictions.py \\
        --input  data/little_things/predictions.csv \\
        --output results/little_things_predictions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rmse(y_true: "array-like", y_pred: "array-like") -> float:
    """Root Mean Square Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _repo_root() -> Path:
    """Absolute path to the repository root (parent of *scripts/*)."""
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze(in_csv: Path, out_dir: Path) -> dict:
    """Run the full analysis pipeline and return the summary dictionary."""
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)

    required = [
        "galaxy_id",
        "logVobs",
        "logV_btfr",
        "logV_interp",
        "residual_btfr",
        "residual_interp",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    # ------------------------------------------------------------------
    # Corrección de offset global del modelo interpolado
    #
    #   residual = obs − pred
    #   Si la media es negativa → el modelo sobrepredice.
    #   Corrección: logV_interp_corr = logV_interp + offset   (offset < 0 → resta)
    # ------------------------------------------------------------------
    offset = float(df["residual_interp"].mean())
    df["logV_interp_corr"] = df["logV_interp"] + offset
    df["residual_interp_corr"] = df["logVobs"] - df["logV_interp_corr"]

    summary: dict = {
        "n_galaxies": int(len(df)),
        "offset_interp_mean": offset,
        "rmse_btfr": rmse(df["logVobs"], df["logV_btfr"]),
        "rmse_interp_raw": rmse(df["logVobs"], df["logV_interp"]),
        "rmse_interp_corr": rmse(df["logVobs"], df["logV_interp_corr"]),
        "std_residual_btfr": float(df["residual_btfr"].std(ddof=1)),
        "std_residual_interp_raw": float(df["residual_interp"].std(ddof=1)),
        "std_residual_interp_corr": float(df["residual_interp_corr"].std(ddof=1)),
        "mean_residual_btfr": float(df["residual_btfr"].mean()),
        "mean_residual_interp_raw": float(df["residual_interp"].mean()),
        "mean_residual_interp_corr": float(df["residual_interp_corr"].mean()),
    }
    summary["delta_rmse_raw_minus_btfr"] = (
        summary["rmse_interp_raw"] - summary["rmse_btfr"]
    )
    summary["delta_rmse_corr_minus_btfr"] = (
        summary["rmse_interp_corr"] - summary["rmse_btfr"]
    )

    # ------------------------------------------------------------------
    # Salidas tabulares
    # ------------------------------------------------------------------
    df.to_csv(out_dir / "predictions_corrected.csv", index=False)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as fh:
        for k, v in summary.items():
            fh.write(f"{k}: {v}\n")

    # ------------------------------------------------------------------
    # Figura 1 – Observadas vs Predichas
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["logVobs"], df["logV_btfr"], label="BTFR", alpha=0.8)
    ax.scatter(df["logVobs"], df["logV_interp_corr"], label="SCM corr", alpha=0.8)

    lo = min(df["logVobs"].min(), df["logV_btfr"].min(), df["logV_interp_corr"].min())
    hi = max(df["logVobs"].max(), df["logV_btfr"].max(), df["logV_interp_corr"].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)

    ax.set_xlabel("log Vobs")
    ax.set_ylabel("log Vpred")
    ax.set_title("Observed vs Predicted Velocity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "observed_vs_predicted.png", dpi=200)
    fig.savefig(out_dir / "observed_vs_predicted.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figura 2 – Distribución de residuales
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["residual_btfr"], bins=10, alpha=0.7, label="BTFR")
    ax.hist(df["residual_interp_corr"], bins=10, alpha=0.7, label="SCM corr")
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "residual_distributions.png", dpi=200)
    fig.savefig(out_dir / "residual_distributions.pdf")
    plt.close(fig)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Framework SCM – análisis de predicciones de velocidad galáctica."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "little_things" / "predictions.csv",
        help="Ruta al CSV de entrada (default: data/little_things/predictions.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "little_things_predictions",
        help="Directorio de salida (default: results/little_things_predictions)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    in_csv: Path = args.input if args.input.is_absolute() else _repo_root() / args.input
    out_dir: Path = args.output if args.output.is_absolute() else _repo_root() / args.output

    summary = analyze(in_csv, out_dir)

    print("Análisis completado.")
    print(f"Entrada : {in_csv}")
    print(f"Salida  : {out_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
