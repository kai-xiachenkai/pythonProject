import os
import argparse
import numpy as np
import pandas as pd
from scipy.integrate import odeint

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge

import joblib

DEFAULT_INPUT_CSV = "all_runs_merged.csv"
DEFAULT_OUTDIR = "beyond_emt_ablation"
DEFAULT_BETA_COL = "beta1"
DEFAULT_CV_SPLITS = 5

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
    need = ["total_trans", "RR", "RB", "RG", "BR", "BB", "BG", "GR", "GB", "GG"]
    ensure_cols(df, need)

    df = df.copy()
    T = df["total_trans"].astype(float)
    T_safe = T.replace(0, np.nan)

    inter = ["RR", "RB", "RG", "BR", "BB", "BG", "GR", "GB", "GG"]
    for c in inter:
        df[f"{c}_share"] = (df[c].astype(float) / T_safe).fillna(0.0)

    P = df[[f"{c}_share" for c in inter]].to_numpy(dtype=float)
    df["interaction_hhi"] = (P ** 2).sum(axis=1)
    df["log_total_trans"] = np.log1p(T.astype(float))
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

def metrics_two_modes(y_true, y_pred, tau):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {}
    out["mae_all"] = float(mean_absolute_error(y_true, y_pred))
    out["r2_all"] = float(r2_score(y_true, y_pred))

    mask = y_true >= tau
    out["n_all"] = int(len(y_true))
    out["n_outbreak"] = int(mask.sum())
    if mask.sum() >= 5:
        out["mae_outbreak"] = float(mean_absolute_error(y_true[mask], y_pred[mask]))
        out["r2_outbreak"] = float(r2_score(y_true[mask], y_pred[mask]))
    else:
        out["mae_outbreak"] = np.nan
        out["r2_outbreak"] = np.nan
    return out

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

def build_boost_model():
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=6,
        max_iter=500,
        random_state=0
    )

def oof_residual_predictions(df, features, model_type, cv_splits=5):
    X = df[features]
    y = df["residual_vs_emt"].astype(float).to_numpy()
    groups = df["scenario_id"].astype(str).to_numpy()

    gkf = GroupKFold(n_splits=cv_splits)
    oof = np.zeros(len(df), dtype=float)

    for tr, te in gkf.split(X, y, groups):
        if model_type == "gam":
            model = build_gam_pipeline(features)
        elif model_type == "boost":
            model = build_boost_model()
        else:
            raise ValueError("model_type must be 'gam' or 'boost'")
        model.fit(X.iloc[tr], y[tr])
        oof[te] = model.predict(X.iloc[te])
    return oof

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default=DEFAULT_INPUT_CSV)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--beta_col", default=DEFAULT_BETA_COL)
    ap.add_argument("--cv_splits", type=int, default=DEFAULT_CV_SPLITS)
    ap.add_argument("--taus", default=os.environ.get("TAUS", "0.01,0.03,0.05,0.10"))
    ap.add_argument("--models", default="gam,boost", help="comma-separated: gam,boost")
    args = ap.parse_args()

    taus = [float(x.strip()) for x in args.taus.split(",") if x.strip()]
    model_types = [x.strip().lower() for x in args.models.split(",") if x.strip()]
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df = add_mixing_features(df)

    cache_csv = os.path.join(args.outdir, "emt_cache.csv")
    df = compute_emt_baseline_cached(df, cache_csv=cache_csv, beta_col=args.beta_col)

    y_true = df["R_final"].astype(float).to_numpy()
    y_emt = df["R_emt_pred"].astype(float).to_numpy()

    feature_sets = [
        ("A0_calibration_only", ["R_emt_pred"]),
        ("A1_plus_log_total_trans", ["R_emt_pred", "log_total_trans"]),
        ("A2_plus_entropy", ["R_emt_pred", "log_total_trans", "interaction_entropy"]),
        ("A3_plus_hhi", ["R_emt_pred", "log_total_trans", "interaction_entropy", "interaction_hhi"]),
        ("A4_plus_cross_trans", ["R_emt_pred", "log_total_trans", "interaction_entropy", "interaction_hhi", "cross_trans_frac"]),
        ("A5_plus_pr", ["R_emt_pred", "log_total_trans", "interaction_entropy", "interaction_hhi", "cross_trans_frac", "pr"]),
    ]

    rows = []

    # Baseline EMT
    for tau in taus:
        m = metrics_two_modes(y_true, y_emt, tau)
        rows.append({"tau": tau, "subset": "EMT_baseline", "model": "EMT", **m})

    for subset_name, feats in feature_sets:
        for mt in model_types:
            oof_res = oof_residual_predictions(df, feats, mt, cv_splits=args.cv_splits)
            y_pred = y_emt + oof_res

            if subset_name == "A4_plus_cross_trans" and mt == "gam":
                out_oof = df[["scenario_id","k","run_id","R_final","R_emt_pred"]].copy()
                out_oof["residual_oof"] = oof_res
                out_oof["R_pred"] = y_pred
                out_oof.to_csv(os.path.join(args.outdir, "oof_predictions_example.csv"), index=False)

            for tau in taus:
                m = metrics_two_modes(y_true, y_pred, tau)
                rows.append({"tau": tau, "subset": subset_name, "model": mt.upper(), **m})

            if subset_name == "A4_plus_cross_trans":
                if mt == "gam":
                    final_model = build_gam_pipeline(feats).fit(df[feats], df["residual_vs_emt"].astype(float))
                    joblib.dump(final_model, os.path.join(args.outdir, "final_gam_A4.joblib"))
                elif mt == "boost":
                    final_model = build_boost_model().fit(df[feats], df["residual_vs_emt"].astype(float))
                    joblib.dump(final_model, os.path.join(args.outdir, "final_boost_A4.joblib"))

    out = pd.DataFrame(rows)
    out_path = os.path.join(args.outdir, "ablation_mixing_vs_tau.csv")
    out.to_csv(out_path, index=False)
    print("\nSaved:", out_path)
    print(out.sort_values(["tau","model","subset"]).to_string(index=False))

    # Plot: R2_outbreak vs subset for each tau
    try:
        import matplotlib.pyplot as plt
        sub_order = [s for s,_ in feature_sets]
        for mt in model_types:
            plt.figure(figsize=(9,4.5))
            for tau in taus:
                d = out[(out["model"]==mt.upper()) & (out["subset"].isin(sub_order)) & (out["tau"]==tau)].copy()
                d["subset"] = pd.Categorical(d["subset"], categories=sub_order, ordered=True)
                d = d.sort_values("subset")
                plt.plot(range(len(d)), d["r2_outbreak"], marker="o", label=f"tau={tau:g}")
            plt.xticks(range(len(sub_order)), sub_order, rotation=25, ha="right")
            plt.ylim(-5, 1.02)
            plt.title(f"Ablation (mixing closure): R²_outbreak vs feature set [{mt.upper()}]")
            plt.ylabel("R²_outbreak")
            plt.tight_layout()
            fig_path = os.path.join(args.outdir, f"ablation_r2_outbreak_{mt}.png")
            plt.savefig(fig_path, dpi=200)
            plt.close()
            print("Saved:", fig_path)
    except Exception as e:
        print("[WARN] Plotting skipped:", e)

if __name__ == "__main__":
    main()
