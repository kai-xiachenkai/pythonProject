import os
import sys
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


# Optional plots (requires matplotlib); script works without it
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# =========================
# HARD-CODED SETTINGS
# =========================
INPUT_CSV = "all_runs_merged.csv"
OUTDIR = "agg_param_forecast_out"
BETA_COL = "beta1"
TAU = 0.05


# =========================
# Tau sweep (optional)
# =========================
def parse_taus(argv, default_tau):
    """Parse --tau or --taus from CLI or TAUS env var (comma-separated)."""
    taus = None
    tau = None
    for i, a in enumerate(list(argv)):
        if a == "--tau" and i + 1 < len(argv):
            try:
                tau = float(argv[i + 1])
            except Exception:
                pass
        if a == "--taus" and i + 1 < len(argv):
            raw = str(argv[i + 1])
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            try:
                taus = [float(p) for p in parts]
            except Exception:
                taus = None

    if taus is None:
        env = os.environ.get("TAUS", "").strip()
        if env:
            parts = [p.strip() for p in env.split(",") if p.strip()]
            try:
                taus = [float(p) for p in parts]
            except Exception:
                taus = None

    if taus is not None and len(taus) > 0:
        return taus
    if tau is not None:
        return [tau]
    return [float(default_tau)]


def save_tau_plot(metrics_vs_tau, outdir, task_prefix, ycol="R2"):
    """Plot ycol vs tau for rows whose model label starts with task_prefix."""
    if not HAS_MPL:
        return
    dfp = metrics_vs_tau.copy()
    dfp["tau"] = dfp["tau"].astype(float)
    dfp = dfp[dfp["model"].astype(str).str.startswith(task_prefix)].copy()
    if len(dfp) == 0:
        return
    dfp = dfp.sort_values("tau")
    plt.figure()
    for model in dfp["model"].unique():
        sub = dfp[dfp["model"] == model]
        plt.plot(sub["tau"], sub[ycol], marker="o", label=model.replace(task_prefix, "").strip())
    plt.xlabel("tau")
    plt.ylabel(ycol)
    plt.title(f"{task_prefix} {ycol} vs tau (GroupKFold by scenario_id)")
    plt.legend()
    plt.tight_layout()
    safe = task_prefix.replace(":", "").replace("|", "_").replace(" ", "_")
    plt.savefig(os.path.join(outdir, f"{safe}_{ycol}_vs_tau.png"), dpi=200)
    plt.close()
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

    taus = parse_taus(sys.argv[1:], TAU)
    print("taus =", ", ".join(str(float(t)) for t in taus))

    df = pd.read_csv(INPUT_CSV)

    # required per-run columns
    ensure_cols(df, ["scenario_id","k","R_final",
                     "pr","pb","v_fr","v_fb","v_fg","Frb","Frg","Fbg",
                     "v_rr","v_bb","v_gg","gamma",BETA_COL,"n"])

    # ensure EMT exists (cached)
    cache_csv = os.path.join(OUTDIR, "emt_cache.csv")
    df = compute_emt_baseline_cached(df, cache_csv=cache_csv, beta_col=BETA_COL)

    # -------- Tau-independent aggregation to (scenario_id, k) --------
    # Build a base aggregate once, then add tau-specific targets inside loop.
    grp = []
    for (sid, k), g in df.groupby(["scenario_id", "k"]):
        R = g["R_final"].astype(float).to_numpy()

        grp.append({
            "scenario_id": sid,
            "k": float(k),
            "n_runs": int(len(g)),
            "mean_R_all": float(np.mean(R)),
            # keep full R list for tau-dependent recomputation (temporary)
            "_R_vals": R,

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

    agg_base = pd.DataFrame(grp)

    # save a tau-independent base file (without _R_vals)
    agg_base.drop(columns=["_R_vals"]).to_csv(os.path.join(OUTDIR, "agg_by_scenario_k.csv"), index=False)
    print("Saved:", os.path.join(OUTDIR, "agg_by_scenario_k.csv"))

    # -------- Features (parameter-only; forecastable) --------
    param_feats = ["R_emt_pred","k","pr","pb",
                   "v_fr","v_fb","v_fg",
                   "Frb","Frg","Fbg",
                   "v_rr","v_bb","v_gg",
                   "gamma", BETA_COL]

    X_base = agg_base[param_feats].replace([np.inf, -np.inf], np.nan)
    groups_base = agg_base["scenario_id"].astype(str)

    # Validate non-null feature rows once
    good = ~X_base.isna().any(axis=1)
    if not np.all(good):
        agg_base = agg_base.loc[good].reset_index(drop=True)
        X_base = X_base.loc[good].reset_index(drop=True)
        groups_base = groups_base.loc[good].reset_index(drop=True)

    # Shared models (re-fit inside OOF each tau)
    spl_cal = Pipeline([
        ("ct", ColumnTransformer(
            [("spl", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["R_emt_pred"])],
            remainder="drop"
        )),
        ("ridge", Ridge(alpha=0.5))
    ])

    gam_p = Pipeline([
        ("ct", ColumnTransformer(
            [("spl", SplineTransformer(n_knots=SPLINE_KNOTS, degree=SPLINE_DEGREE, include_bias=False), param_feats)],
            remainder="drop"
        )),
        ("ridge", Ridge(alpha=RIDGE_ALPHA))
    ])

    bst_p = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=500, random_state=42)
    bst_s = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=600, random_state=42)

    metrics_rows = []

    # optional: save OOF only for single tau (to avoid writing many files unless you want that)
    save_single_tau_oof = (len(taus) == 1)
    outA_single = None
    outB_single = None

    for tau in taus:
        tau = float(tau)

        # -------- Build tau-specific targets on top of agg_base --------
        agg = agg_base.copy()

        p_outbreak_list = []
        mean_R_cond_list = []
        median_R_cond_list = []

        for R in agg["_R_vals"]:
            R = np.asarray(R, dtype=float)
            mask = R >= tau

            p_outbreak_list.append(float(np.mean(mask)))
            if np.any(mask):
                mean_R_cond_list.append(float(np.mean(R[mask])))
                median_R_cond_list.append(float(np.median(R[mask])))
            else:
                mean_R_cond_list.append(np.nan)
                median_R_cond_list.append(np.nan)

        agg["p_outbreak"] = p_outbreak_list
        agg["mean_R_cond"] = mean_R_cond_list
        agg["median_R_cond"] = median_R_cond_list

        X = agg[param_feats].replace([np.inf, -np.inf], np.nan)
        groups = agg["scenario_id"].astype(str)

        # -------- TASK A: Outbreak probability p_outbreak --------
        y_p = agg["p_outbreak"].astype(float)

        # Baseline A1: EMT-threshold (tau-specific)
        p_hat_emt_hard = (agg["R_emt_pred"].astype(float) >= tau).astype(float)

        # A2: EMT-calibrated spline
        X_cal = agg[["R_emt_pred"]]
        oof_p_cal = oof_groupkfold(spl_cal, X_cal, y_p, groups, n_splits=CV_SPLITS)
        oof_p_cal = np.clip(oof_p_cal, 0, 1)

        # A3: Param+EMT GAM
        oof_p_gam = oof_groupkfold(gam_p, X, y_p, groups, n_splits=CV_SPLITS)
        oof_p_gam = np.clip(oof_p_gam, 0, 1)

        # A4: Param+EMT Boosting
        oof_p_bst = oof_groupkfold(bst_p, X, y_p, groups, n_splits=CV_SPLITS)
        oof_p_bst = np.clip(oof_p_bst, 0, 1)

        # -------- TASK B: Conditional severity mean_R_cond --------
        mask_cond = ~agg["mean_R_cond"].isna()
        agg_cond = agg.loc[mask_cond].reset_index(drop=True)

        # If no outbreak groups at this tau, skip severity metrics safely
        if len(agg_cond) > 0:
            Xc = agg_cond[param_feats].replace([np.inf, -np.inf], np.nan)
            gc = agg_cond["scenario_id"].astype(str)
            y_s = agg_cond["mean_R_cond"].astype(float)

            # Baseline B1: EMT directly
            s_hat_emt = agg_cond["R_emt_pred"].astype(float).to_numpy()

            # B2/B3
            oof_s_gam = oof_groupkfold(gam_p, Xc, y_s, gc, n_splits=CV_SPLITS)
            oof_s_bst = oof_groupkfold(bst_s, Xc, y_s, gc, n_splits=CV_SPLITS)

            sev_rows = [
                report_metrics("Severity|outbreak: EMT", y_s, s_hat_emt),
                report_metrics("Severity|outbreak: Param+EMT GAM-splines", y_s, oof_s_gam),
                report_metrics("Severity|outbreak: Param+EMT Boosting", y_s, oof_s_bst),
            ]
        else:
            y_s = None
            s_hat_emt = None
            oof_s_gam = None
            oof_s_bst = None
            sev_rows = []

        # -------- Collect metrics for this tau --------
        prob_rows = [
            report_metrics("P(outbreak): EMT-threshold", y_p, p_hat_emt_hard),
            report_metrics("P(outbreak): EMT-calibrated spline", y_p, oof_p_cal),
            report_metrics("P(outbreak): Param+EMT GAM-splines", y_p, oof_p_gam),
            report_metrics("P(outbreak): Param+EMT Boosting", y_p, oof_p_bst),
        ]

        for r in (prob_rows + sev_rows):
            metrics_rows.append({"tau": tau, **r})

        # Save OOF prediction tables only in single-tau mode (backward-compatible behavior)
        if save_single_tau_oof:
            outA = agg.drop(columns=["_R_vals"]).copy()
            outA["p_hat_emt_threshold"] = p_hat_emt_hard
            outA["p_hat_emt_cal_spline_oof"] = oof_p_cal
            outA["p_hat_param_gam_oof"] = oof_p_gam
            outA["p_hat_param_boost_oof"] = oof_p_bst
            outA_single = outA

            if len(agg_cond) > 0:
                outB = agg_cond.drop(columns=["_R_vals"]).copy()
                outB["sev_hat_emt"] = s_hat_emt
                outB["sev_hat_param_gam_oof"] = oof_s_gam
                outB["sev_hat_param_boost_oof"] = oof_s_bst
                outB_single = outB
            else:
                outB_single = agg_cond.drop(columns=["_R_vals"]).copy()

    # -------- Save metrics --------
    metrics_vs_tau = pd.DataFrame(metrics_rows)
    metrics_vs_tau_path = os.path.join(OUTDIR, "metrics_agg_param_forecast_vs_tau.csv")
    metrics_vs_tau.to_csv(metrics_vs_tau_path, index=False)

    # backward compatible single-tau file
    if len(taus) == 1:
        metrics_path = os.path.join(OUTDIR, "metrics_agg_param_forecast.csv")
        metrics_vs_tau.drop(columns=["tau"]).to_csv(metrics_path, index=False)

    print("\n=== Aggregated forecast from parameters (GroupKFold by scenario_id) ===")
    print("taus =", ", ".join(str(float(t)) for t in taus))
    print(metrics_vs_tau.to_string(index=False))
    print("\nSaved:", metrics_vs_tau_path)

    # -------- Save OOF predictions (single tau only) --------
    if save_single_tau_oof and outA_single is not None:
        outA_path = os.path.join(OUTDIR, "agg_with_oof_prob_predictions.csv")
        outA_single.to_csv(outA_path, index=False)
        print("Saved:", outA_path)

    if save_single_tau_oof and outB_single is not None:
        outB_path = os.path.join(OUTDIR, "agg_with_oof_severity_predictions.csv")
        outB_single.to_csv(outB_path, index=False)
        print("Saved:", outB_path)

    # -------- Optional tau plots --------
    # Note: report_metrics returns columns MAE/R2/RMSE (uppercase)
    if HAS_MPL and len(taus) > 1:
        save_tau_plot(metrics_vs_tau, OUTDIR, "P(outbreak):", ycol="R2")
        save_tau_plot(metrics_vs_tau, OUTDIR, "P(outbreak):", ycol="MAE")
        save_tau_plot(metrics_vs_tau, OUTDIR, "Severity|outbreak:", ycol="R2")
        save_tau_plot(metrics_vs_tau, OUTDIR, "Severity|outbreak:", ycol="MAE")


if __name__ == "__main__":
    main()
