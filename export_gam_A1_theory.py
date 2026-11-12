import os
import argparse
import numpy as np
import pandas as pd
from scipy.integrate import odeint

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
import joblib

DEFAULT_INPUT_CSV = "all_runs_merged.csv"
DEFAULT_OUTDIR = "beyond_emt_ablation"

SPLINE_KNOTS = 8
SPLINE_DEGREE = 3
RIDGE_ALPHA = 1.0


def smodel(s, t,
           v_fr, v_fb, v_fg,
           v_rr, v_rb, v_rg, v_br, v_bb, v_bg, v_gr, v_gb, v_gg,
           p_r, p_b, p_g,
           beta, gamma, n):
    eps = 1e-12
    vc = (
        v_rr * (p_r ** 2) + v_bb * (p_b ** 2) + v_gg * (p_g ** 2)
        + v_rb * p_r * p_b + v_br * p_b * p_r
        + v_bg * p_b * p_g + v_gb * p_g * p_b
        + v_rg * p_r * p_g + v_gr * p_g * p_r
    )
    vf = v_fr * p_r + v_fb * p_b + v_fg * p_g
    p = vc / (vc + n * vf + eps)

    s_safe = max(float(s), eps)
    c = n - (gamma / (beta * p + eps)) * np.log(n - 1)

    dsdt = -gamma * s_safe * np.log(s_safe) + beta * p * (s_safe ** 2) - beta * p * c * s_safe
    return dsdt


def EMT1(v_fr, v_fb, v_fg,
         v_rr, v_rb, v_rg, v_br, v_bb, v_bg, v_gr, v_gb, v_gg,
         p_r, p_b, p_g,
         beta, gamma, n):
    s0 = int(n) - 1
    t = np.linspace(0, 1500, 1500)
    S_t = odeint(
        smodel, s0, t,
        args=(v_fr, v_fb, v_fg,
              v_rr, v_rb, v_rg, v_br, v_bb, v_bg, v_gr, v_gb, v_gg,
              p_r, p_b, p_g,
              beta, gamma, int(n))
    )
    S_final = float(np.ravel(S_t)[-1])
    R_final_count = int(n) - S_final
    return R_final_count


def ensure_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def derive_inter_coalescence_from_F(df):
    ensure_cols(df, ["v_rr", "Frb", "Frg", "Fbg"])
    df = df.copy()
    df["v_rb"] = df["v_rr"] * df["Frb"]
    df["v_br"] = df["v_rb"]
    df["v_rg"] = df["v_rr"] * df["Frg"]
    df["v_gr"] = df["v_rg"]
    df["v_bg"] = df["v_rr"] * df["Fbg"]
    df["v_gb"] = df["v_bg"]
    return df


def add_mixing_features(df):
    ensure_cols(df, ["total_trans"])
    df = df.copy()
    df["log_total_trans"] = np.log1p(df["total_trans"].astype(float))
    return df


def compute_emt_baseline_cached(df, cache_csv, beta_col):
    need = ["scenario_id", "k", "pr", "pb", "n",
            "v_fr", "v_fb", "v_fg", "v_rr", "v_bb", "v_gg", "gamma", beta_col,
            "Frb", "Frg", "Fbg"]
    ensure_cols(df, need)

    df = df.copy()
    df["p_g"] = 1.0 - df["pr"].astype(float) - df["pb"].astype(float)
    df = derive_inter_coalescence_from_F(df)

    key = ["scenario_id", "k"]
    uniq_cols = key + [
        "v_fr", "v_fb", "v_fg",
        "v_rr", "v_rb", "v_rg", "v_br", "v_bb", "v_bg", "v_gr", "v_gb", "v_gg",
        "pr", "pb", "p_g",
        beta_col, "gamma", "n"
    ]
    uniq = df[uniq_cols].drop_duplicates().copy()

    if os.path.exists(cache_csv):
        cache = pd.read_csv(cache_csv)
        if set(["scenario_id", "k", "R_emt_pred"]).issubset(cache.columns):
            uniq = uniq.merge(cache[["scenario_id", "k", "R_emt_pred"]], on=key, how="left")
        else:
            uniq["R_emt_pred"] = np.nan
    else:
        uniq["R_emt_pred"] = np.nan

    todo = uniq["R_emt_pred"].isna()
    if todo.any():
        def row_emt(r):
            R_cnt = EMT1(
                r.v_fr, r.v_fb, r.v_fg,
                r.v_rr, r.v_rb, r.v_rg, r.v_br, r.v_bb, r.v_bg, r.v_gr, r.v_gb, r.v_gg,
                r.pr, r.pb, r.p_g,
                getattr(r, beta_col), r.gamma, int(r.n)
            )
            return float(R_cnt) / float(r.n)

        uniq.loc[todo, "R_emt_pred"] = uniq.loc[todo].apply(row_emt, axis=1)
        uniq[key + ["R_emt_pred"]].drop_duplicates().to_csv(cache_csv, index=False)

    df = df.merge(uniq[key + ["R_emt_pred"]], on=key, how="left")
    df["residual_vs_emt"] = df["R_final"].astype(float) - df["R_emt_pred"].astype(float)
    return df


def build_gam_pipeline(features):
    ct = ColumnTransformer(
        transformers=[
            ("spl", SplineTransformer(
                n_knots=SPLINE_KNOTS,
                degree=SPLINE_DEGREE,
                include_bias=False
            ), features)
        ],
        remainder="drop"
    )
    pipe = Pipeline([
        ("ct", ct),
        ("ridge", Ridge(alpha=RIDGE_ALPHA))
    ])
    return pipe


def export_gam_theory(fitted_pipe, feature_names, out_npz):
    # Export a symbolic representation of the GAM:
    #   y = intercept + sum_j sum_m a_{j,m} * B_{j,m}(x_j)
    # where B_{j,m} are the spline basis functions used by sklearn's SplineTransformer.
    ct = fitted_pipe.named_steps["ct"]
    ridge = fitted_pipe.named_steps["ridge"]
    spl = ct.named_transformers_["spl"]

    knots = [np.array(bs.t, dtype=float) for bs in spl.bsplines_]
    degree = int(spl.bsplines_[0].k)

    n_features = len(feature_names)
    n_out = int(spl.n_features_out_)
    if n_out % n_features != 0:
        raise ValueError(f"Unexpected spline output dim {n_out} for {n_features} features.")
    n_splines_per_feature = n_out // n_features

    coef = ridge.coef_.astype(float).reshape(-1)
    intercept = float(ridge.intercept_)

    np.savez(
        out_npz,
        feature_names=np.array(feature_names, dtype=object),
        degree=np.array(degree, dtype=int),
        n_splines_per_feature=np.array(n_splines_per_feature, dtype=int),
        knots_=np.array(knots, dtype=object),
        coef_=coef,
        intercept_=np.array(intercept, dtype=float),
        include_bias=np.array(False, dtype=bool),
        n_knots=np.array(SPLINE_KNOTS, dtype=int),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default=DEFAULT_INPUT_CSV)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--beta_col", default="beta1")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df = add_mixing_features(df)

    cache_csv = os.path.join(args.outdir, "emt_cache.csv")
    df = compute_emt_baseline_cached(df, cache_csv=cache_csv, beta_col=args.beta_col)

    feats = ["R_emt_pred", "log_total_trans"]  # A1 feature set
    X = df[feats]
    y = df["residual_vs_emt"].astype(float).to_numpy()

    model = build_gam_pipeline(feats)
    model.fit(X, y)

    joblib_path = os.path.join(args.outdir, "final_gam_A1.joblib")
    joblib.dump(model, joblib_path)

    npz_path = os.path.join(args.outdir, "gam_A1_theory.npz")
    export_gam_theory(model, feats, npz_path)

    print("Saved:", joblib_path)
    print("Saved:", npz_path)
    print("Print the symbolic equation with:")
    print("  python gam_symbolic_A1.py --npz", npz_path)


if __name__ == "__main__":
    main()
