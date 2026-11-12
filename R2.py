"""
Partial R² decomposition for the A1 GAM closure.

Run this from the same directory as export_gam_A1_theory.py and all_runs_merged.csv.
It fits three models on the same data:
  (1) Full model:     Delta ~ f_R(R_emt) + f_A(A)
  (2) R_emt only:     Delta ~ f_R(R_emt)
  (3) A only:         Delta ~ f_A(A)

and reports the R² for each, plus the incremental gain from adding A on top of R_emt.

Usage:
    python partial_r2_decomposition.py
    python partial_r2_decomposition.py --input_csv my_data.csv --beta_col beta1
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# ── same settings as export_gam_A1_theory.py ────────────────────────────────
SPLINE_KNOTS  = 8
SPLINE_DEGREE = 3
RIDGE_ALPHA   = 1.0

DEFAULT_INPUT_CSV = "all_runs_merged.csv"
DEFAULT_OUTDIR    = "beyond_emt_ablation"


def build_gam_pipeline(features):
    ct = ColumnTransformer(
        transformers=[("spl", SplineTransformer(
            n_knots=SPLINE_KNOTS, degree=SPLINE_DEGREE, include_bias=False
        ), features)],
        remainder="drop"
    )
    return Pipeline([("ct", ct), ("ridge", Ridge(alpha=RIDGE_ALPHA))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default=DEFAULT_INPUT_CSV)
    ap.add_argument("--outdir",    default=DEFAULT_OUTDIR)
    ap.add_argument("--beta_col",  default="beta1")
    args = ap.parse_args()

    # ── Load data (re-use cached EMT predictions if available) ──────────────
    cache_csv = os.path.join(args.outdir, "emt_cache.csv")
    if not os.path.exists(cache_csv):
        raise FileNotFoundError(
            f"EMT cache not found at {cache_csv}. "
            "Run export_gam_A1_theory.py first to generate it."
        )

    df = pd.read_csv(args.input_csv)
    df["log_total_trans"] = np.log1p(df["total_trans"].astype(float))
    df["p_g"] = 1.0 - df["pr"].astype(float) - df["pb"].astype(float)

    cache = pd.read_csv(cache_csv)
    df = df.merge(cache[["scenario_id", "k", "R_emt_pred"]], on=["scenario_id", "k"], how="left")
    df["residual_vs_emt"] = df["R_final"].astype(float) - df["R_emt_pred"].astype(float)
    df = df.dropna(subset=["R_emt_pred", "residual_vs_emt", "log_total_trans"])

    y = df["residual_vs_emt"].to_numpy()

    # ── Fit three models ─────────────────────────────────────────────────────
    models = {
        "Full: f_R(R_emt) + f_A(A)": ["R_emt_pred", "log_total_trans"],
        "R_emt only: f_R(R_emt)":    ["R_emt_pred"],
        "A only:     f_A(A)":        ["log_total_trans"],
    }

    results = {}
    for label, feats in models.items():
        pipe = build_gam_pipeline(feats)
        pipe.fit(df[feats], y)
        y_pred = pipe.predict(df[feats])
        r2 = r2_score(y, y_pred)
        results[label] = r2
        print(f"{label:<40s}  R² = {r2:.4f}")

    # ── Incremental gain ─────────────────────────────────────────────────────
    r2_R   = results["R_emt only: f_R(R_emt)"]
    r2_A   = results["A only:     f_A(A)"]
    r2_full = results["Full: f_R(R_emt) + f_A(A)"]

    print()
    print(f"Incremental R² from adding A on top of R_emt:  {r2_full - r2_R:+.4f}")
    print(f"Incremental R² from adding R_emt on top of A:  {r2_full - r2_A:+.4f}")
    print()
    print("── Copy these numbers into the thesis ──────────────────────────────")
    print(f"  Full model R²         = {r2_full:.3f}")
    print(f"  f_R alone R²          = {r2_R:.3f}")
    print(f"  f_A alone R²          = {r2_A:.3f}")
    print(f"  Gain from f_A         = {r2_full - r2_R:+.3f}")


if __name__ == "__main__":
    main()