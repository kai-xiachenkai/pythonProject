import os
import numpy as np
import pandas as pd
from scipy.integrate import odeint

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge

# =========================
# HARD-CODED SETTINGS
# =========================
INPUT_CSV = "all_runs_merged.csv"
OUTDIR = "agg_param_forecast_out"
BETA_COL = "beta1"
TAU = 0.05
CV_SPLITS = 5

# GAM (spline ridge) settings
SPLINE_KNOTS = 7
SPLINE_DEGREE = 3
RIDGE_ALPHA = 1.0


# =========================
# EMT reduced ODE (your model)
# =========================
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
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"Missing required columns: {miss}")


def derive_inter_coalescence_from_F(df):
    # derive v_rb etc from v_rr and Frb/Frg/Fbg
    ensure_cols(df, ["v_rr", "Frb", "Frg", "Fbg"])
    df = df.copy()
    df["v_rb"] = df["v_rr"] * df["Frb"]
    df["v_br"] = df["v_rb"]
    df["v_rg"] = df["v_rr"] * df["Frg"]
    df["v_gr"] = df["v_rg"]
    df["v_bg"] = df["v_rr"] * df["Fbg"]
    df["v_gb"] = df["v_bg"]
    return df


def compute_emt_baseline_cached(df, cache_csv, beta_col=BETA_COL):
    need = ["scenario_id","k","pr","pb","n",
            "v_fr","v_fb","v_fg","v_rr","v_bb","v_gg","gamma", beta_col,
            "Frb","Frg","Fbg"]
    ensure_cols(df, need)

    df = df.copy()
    df["p_g"] = 1.0 - df["pr"].astype(float) - df["pb"].astype(float)
    df = derive_inter_coalescence_from_F(df)

    key = ["scenario_id", "k"]
    uniq_cols = key + [
        "v_fr","v_fb","v_fg",
        "v_rr","v_rb","v_rg","v_br","v_bb","v_bg","v_gr","v_gb","v_gg",
        "pr","pb","p_g",
        beta_col,"gamma","n"
    ]
    uniq = df[uniq_cols].drop_duplicates().copy()

    # load cache if exists
    if os.path.exists(cache_csv):
        cache = pd.read_csv(cache_csv)
        if set(["scenario_id","k","R_emt_pred"]).issubset(cache.columns):
            uniq = uniq.merge(cache[["scenario_id","k","R_emt_pred"]], on=key, how="left")
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
    return df


def oof_groupkfold(model, X, y, groups, n_splits=5):
    groups = np.asarray(groups)
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y), dtype=float)
    for tr, te in gkf.split(X, y, groups=groups):
        model.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = model.predict(X.iloc[te])
    return oof


def report_metrics(name, y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "model": name,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "n": int(len(y_true))
    }


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Cannot find {INPUT_CSV} in current folder.")

    df = pd.read_csv(INPUT_CSV)

    # required per-run columns
    ensure_cols(df, ["scenario_id","k","R_final",
                     "pr","pb","v_fr","v_fb","v_fg","Frb","Frg","Fbg",
                     "v_rr","v_bb","v_gg","gamma",BETA_COL,"n"])

    # ensure EMT exists (cached)
    cache_csv = os.path.join(OUTDIR, "emt_cache.csv")
    df = compute_emt_baseline_cached(df, cache_csv=cache_csv, beta_col=BETA_COL)

    # -------- Aggregate to (scenario_id, k) --------
    df["outbreak"] = (df["R_final"].astype(float) >= TAU).astype(int)

    def cond_mean(series, mask):
        vals = series[mask]
        return np.nan if len(vals) == 0 else float(np.mean(vals))

    def cond_median(series, mask):
        vals = series[mask]
        return np.nan if len(vals) == 0 else float(np.median(vals))

    grp = []
    for (sid, k), g in df.groupby(["scenario_id","k"]):
        R = g["R_final"].astype(float).to_numpy()
        mask = R >= TAU
        grp.append({
            "scenario_id": sid,
            "k": float(k),
            "n_runs": int(len(g)),
            "p_outbreak": float(np.mean(mask)),
            "mean_R_cond": cond_mean(pd.Series(R), mask),
            "median_R_cond": cond_median(pd.Series(R), mask),
            "mean_R_all": float(np.mean(R)),
            # parameter snapshot (constant within group)
            "R_emt_pred": float(g["R_emt_pred"].iloc[0]),
            "pr": float(g["pr"].iloc[0]),
            "pb": float(g["pb"].iloc[0]),
            "v_fr": float(g["v_fr"].iloc[0]),
            "v_fb": float(g["v_fb"].iloc[0]),
            "v_fg": float(g["v_fg"].iloc[0]),
            "Frb": float(g["Frb"].iloc[0]),
            "Frg": float(g["Frg"].iloc[0]),
            "Fbg": float(g["Fbg"].iloc[0]),
            "v_rr": float(g["v_rr"].iloc[0]),
            "v_bb": float(g["v_bb"].iloc[0]),
            "v_gg": float(g["v_gg"].iloc[0]),
            "gamma": float(g["gamma"].iloc[0]),
            BETA_COL: float(g[BETA_COL].iloc[0]),
            "n": float(g["n"].iloc[0]),
        })

    agg = pd.DataFrame(grp)
    agg.to_csv(os.path.join(OUTDIR, "agg_by_scenario_k.csv"), index=False)
    print("Saved:", os.path.join(OUTDIR, "agg_by_scenario_k.csv"))

    # -------- Features (parameter-only; forecastable) --------
    param_feats = ["R_emt_pred","k","pr","pb",
                   "v_fr","v_fb","v_fg",
                   "Frb","Frg","Fbg",
                   "v_rr","v_bb","v_gg",
                   "gamma", BETA_COL]

    X = agg[param_feats].replace([np.inf, -np.inf], np.nan)
    groups = agg["scenario_id"].astype(str)

    # -------- TASK A: Outbreak probability p_outbreak --------
    y_p = agg["p_outbreak"].astype(float)

    # Baseline A1: EMT-threshold (hard, uncalibrated)
    p_hat_emt_hard = (agg["R_emt_pred"].astype(float) >= TAU).astype(float)

    # Model A2: EMT-calibrated spline (only uses R_emt_pred)
    spl_cal = Pipeline([
        ("ct", ColumnTransformer(
            [("spl", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["R_emt_pred"])],
            remainder="drop"
        )),
        ("ridge", Ridge(alpha=0.5))
    ])
    X_cal = agg[["R_emt_pred"]]
    oof_p_cal = oof_groupkfold(spl_cal, X_cal, y_p, groups, n_splits=CV_SPLITS)
    oof_p_cal = np.clip(oof_p_cal, 0, 1)

    # Model A3: Param+EMT GAM (splines on all param features)
    gam_p = Pipeline([
        ("ct", ColumnTransformer(
            [("spl", SplineTransformer(n_knots=SPLINE_KNOTS, degree=SPLINE_DEGREE, include_bias=False), param_feats)],
            remainder="drop"
        )),
        ("ridge", Ridge(alpha=RIDGE_ALPHA))
    ])
    oof_p_gam = oof_groupkfold(gam_p, X, y_p, groups, n_splits=CV_SPLITS)
    oof_p_gam = np.clip(oof_p_gam, 0, 1)

    # Model A4: Param+EMT Boosting
    bst_p = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=500, random_state=42)
    oof_p_bst = oof_groupkfold(bst_p, X, y_p, groups, n_splits=CV_SPLITS)
    oof_p_bst = np.clip(oof_p_bst, 0, 1)

    # -------- TASK B: Conditional severity mean_R_cond --------
    # only evaluate groups that have at least one outbreak run
    mask_cond = ~agg["mean_R_cond"].isna()
    agg_cond = agg.loc[mask_cond].reset_index(drop=True)
    Xc = agg_cond[param_feats].replace([np.inf, -np.inf], np.nan)
    gc = agg_cond["scenario_id"].astype(str)
    y_s = agg_cond["mean_R_cond"].astype(float)

    # Baseline B1: EMT directly
    s_hat_emt = agg_cond["R_emt_pred"].astype(float).to_numpy()

    # Model B2: Param+EMT GAM
    oof_s_gam = oof_groupkfold(gam_p, Xc, y_s, gc, n_splits=CV_SPLITS)

    # Model B3: Param+EMT Boosting
    bst_s = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=600, random_state=42)
    oof_s_bst = oof_groupkfold(bst_s, Xc, y_s, gc, n_splits=CV_SPLITS)

    # -------- Report --------
    rows = []

    rows.append(report_metrics("P(outbreak): EMT-threshold", y_p, p_hat_emt_hard))
    rows.append(report_metrics("P(outbreak): EMT-calibrated spline", y_p, oof_p_cal))
    rows.append(report_metrics("P(outbreak): Param+EMT GAM-splines", y_p, oof_p_gam))
    rows.append(report_metrics("P(outbreak): Param+EMT Boosting", y_p, oof_p_bst))

    rows.append(report_metrics("Severity|outbreak: EMT", y_s, s_hat_emt))
    rows.append(report_metrics("Severity|outbreak: Param+EMT GAM-splines", y_s, oof_s_gam))
    rows.append(report_metrics("Severity|outbreak: Param+EMT Boosting", y_s, oof_s_bst))

    metrics = pd.DataFrame(rows)
    metrics_path = os.path.join(OUTDIR, "metrics_agg_param_forecast.csv")
    metrics.to_csv(metrics_path, index=False)

    print("\n=== Aggregated forecast from parameters (GroupKFold by scenario_id) ===")
    print(f"tau = {TAU}")
    print(metrics.to_string(index=False))
    print("\nSaved:", metrics_path)

    # Save OOF predictions for plotting
    outA = agg.copy()
    outA["p_hat_emt_th" \
         "reshold"] = p_hat_emt_hard
    outA["p_hat_emt_cal_spline_oof"] = oof_p_cal
    outA["p_hat_param_gam_oof"] = oof_p_gam
    outA["p_hat_param_boost_oof"] = oof_p_bst
    outA_path = os.path.join(OUTDIR, "agg_with_oof_prob_predictions.csv")
    outA.to_csv(outA_path, index=False)
    print("Saved:", outA_path)

    outB = agg_cond.copy()
    outB["sev_hat_emt"] = s_hat_emt
    outB["sev_hat_param_gam_oof"] = oof_s_gam
    outB["sev_hat_param_boost_oof"] = oof_s_bst
    outB_path = os.path.join(OUTDIR, "agg_with_oof_severity_predictions.csv")
    outB.to_csv(outB_path, index=False)
    print("Saved:", outB_path)


if __name__ == "__main__":
    main()
