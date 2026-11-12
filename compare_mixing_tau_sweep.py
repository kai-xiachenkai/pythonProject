import os
import sys
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

# =========================
# HARD-CODED SETTINGS
# =========================
INPUT_CSV = "all_runs_merged.csv"     # must exist next to this file (or set absolute path)
OUTDIR = "beyond_emt_results_param_vs_mixing"
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


def save_tau_summary_plot(metrics_vs_tau, outdir, ycol="r2_outbreak", filename="r2_outbreak_vs_tau.png"):
    if not HAS_MPL:
        return
    dfp = metrics_vs_tau.copy()
    dfp["tau"] = dfp["tau"].astype(float)
    dfp = dfp.sort_values("tau")
    plt.figure()
    for model in dfp["model"].unique():
        sub = dfp[dfp["model"] == model]
        plt.plot(sub["tau"], sub[ycol], marker="o", label=model)
    plt.xlabel("tau")
    plt.ylabel(ycol)
    plt.title(f"{ycol} vs tau (GroupKFold by scenario_id)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=200)
    plt.close()
CV_SPLITS = 5

# GAM spline settings
SPLINE_KNOTS = 8
SPLINE_DEGREE = 3
RIDGE_ALPHA = 1.0

# Optional plots (requires matplotlib); script works without it
MAKE_PLOTS = True
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    MAKE_PLOTS = False


# =========================
# EMT reduced ODE: state = S(t)
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


# =========================
# Helpers / feature engineering
# =========================
def ensure_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


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


def compute_emt_baseline_cached(df, cache_csv, beta_col=BETA_COL):
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


def export_gam_theory(gam_model, gam_feats, outpath_npz):
    """
    Version-robust export (works whether sklearn exposes knots_ or bsplines_).
    Saves an explicit additive-spline residual law:
      Δ = b0 + Σ_j Σ_k a_{j,k} B_{j,k}(x_j)
    """
    ct = gam_model.named_steps["ct"]
    ridge = gam_model.named_steps["ridge"]
    spl = ct.named_transformers_["spl"]

    feature_names = np.array(gam_feats, dtype=object)
    degree = int(getattr(spl, "degree", 3))
    include_bias = bool(getattr(spl, "include_bias", False))

    # Extract knots robustly
    if hasattr(spl, "knots_"):
        knots = spl.knots_.copy()
    elif hasattr(spl, "bsplines_"):
        knot_list = []
        for item in spl.bsplines_:
            if isinstance(item, list):
                knot_list.append(np.array(item[0].t))
            else:
                knot_list.append(np.array(item.t))
        knots = np.array(knot_list, dtype=object)
    else:
        knots = np.array([], dtype=object)

    coef = ridge.coef_.copy()
    intercept = float(ridge.intercept_)

    n_features = len(gam_feats)
    try:
        n_out = int(spl.n_features_out_)
    except Exception:
        n_out = int(spl.transform(np.zeros((1, n_features))).shape[1])

    n_splines_per_feature = int(n_out // n_features) if n_features else 0

    np.savez(
        outpath_npz,
        feature_names=feature_names,
        degree=degree,
        include_bias=include_bias,
        knots_=knots,
        coef_=coef,
        intercept_=intercept,
        n_splines_per_feature=n_splines_per_feature
    )


def scatter_plot(y_true, y_pred, outpath, title):
    if not (MAKE_PLOTS and HAS_MPL):
        return
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.15)
    mn = min(float(np.min(y_true)), float(np.min(y_pred)))
    mx = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("Observed R_final")
    plt.ylabel("Predicted R_final")
    plt.title(title)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


# =========================
# Main
# =========================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = INPUT_CSV if os.path.isabs(INPUT_CSV) else os.path.join(script_dir, INPUT_CSV)
    outdir = OUTDIR if os.path.isabs(OUTDIR) else os.path.join(script_dir, OUTDIR)
    os.makedirs(outdir, exist_ok=True)

    taus = parse_taus(sys.argv[1:], TAU)

    if not os.path.exists(input_path):
        print(f"[ERROR] Cannot find INPUT_CSV at:\n  {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)

    # minimal required columns (based on your header)
    ensure_cols(df, [
        "scenario_id", "k", "run_id", "R_final",
        "RR", "RB", "RG", "BR", "BB", "BG", "GR", "GB", "GG", "total_trans",
        "pr", "pb",
        "v_fr", "v_fb", "v_fg",
        "Frb", "Frg", "Fbg",
        "v_rr", "v_bb", "v_gg",
        "gamma", BETA_COL, "n",
        "cross_trans_frac", "interaction_entropy"
    ])

    # EMT baseline (cached)
    emt_cache_csv = os.path.join(outdir, "emt_cache.csv")
    df = compute_emt_baseline_cached(df, cache_csv=emt_cache_csv, beta_col=BETA_COL)

    # mixing features (shares, hhi, log_total_trans)
    df = add_mixing_features(df)

    # targets
    y_true_R = df["R_final"].astype(float)
    y_resid = df["residual_vs_emt"].astype(float)
    R_emt = df["R_emt_pred"].astype(float)
    groups = df["scenario_id"].astype(str)

    # -------------------------
    # Feature sets
    # -------------------------
    # Parameter-only features (available at time 0 -> forecastable)
    param_feats = [
        "R_emt_pred",
        "k", "pr", "pb",
        "v_fr", "v_fb", "v_fg",
        "Frb", "Frg", "Fbg",
        "v_rr", "v_bb", "v_gg",
        "gamma", BETA_COL
    ]

    # Mixing-closure features (requires infection outcomes -> closure)
    share_feats = [c for c in df.columns if c.endswith("_share")]
    mixing_feats = [
        "R_emt_pred",
        "pr",
        "cross_trans_frac",
        "interaction_entropy",
        "log_total_trans",
        "interaction_hhi",
    ] + sorted(share_feats)

    X_param = df[param_feats].replace([np.inf, -np.inf], np.nan)
    X_mix = df[mixing_feats].replace([np.inf, -np.inf], np.nan)
    X_gam_mix = df[["R_emt_pred", "pr", "cross_trans_frac", "interaction_entropy", "log_total_trans", "interaction_hhi"]].replace(
        [np.inf, -np.inf], np.nan
    )
    gam_mix_feats = list(X_gam_mix.columns)

    good = (
        (~X_param.isna().any(axis=1))
        & (~X_mix.isna().any(axis=1))
        & (~X_gam_mix.isna().any(axis=1))
        & (~y_resid.isna())
        & (~y_true_R.isna())
        & (~R_emt.isna())
    )

    df = df.loc[good].reset_index(drop=True)
    X_param = X_param.loc[good].reset_index(drop=True)
    X_mix = X_mix.loc[good].reset_index(drop=True)
    X_gam_mix = X_gam_mix.loc[good].reset_index(drop=True)
    y_true_R = y_true_R.loc[good].reset_index(drop=True)
    y_resid = y_resid.loc[good].reset_index(drop=True)
    R_emt = R_emt.loc[good].reset_index(drop=True)
    groups = groups.loc[good].reset_index(drop=True)

    # -------------------------
    # Models (predict residual)
    # -------------------------
    # Boosting
    boost_param = HistGradientBoostingRegressor(
        max_depth=6, learning_rate=0.05, max_iter=700, random_state=42
    )
    boost_mix = HistGradientBoostingRegressor(
        max_depth=6, learning_rate=0.05, max_iter=700, random_state=42
    )

    # GAM-like splines
    spline_param = SplineTransformer(
        n_knots=SPLINE_KNOTS, degree=SPLINE_DEGREE, include_bias=False
    )
    gam_param = Pipeline(steps=[
        ("ct", ColumnTransformer([("spl", spline_param, param_feats)], remainder="drop")),
        ("ridge", Ridge(alpha=RIDGE_ALPHA))
    ])

    # for mixing GAM, keep it interpretable (don’t spline 9 shares—too many curves)
    # We spline only the compact mixing stats + EMT + pr.
    spline_mix = SplineTransformer(
        n_knots=SPLINE_KNOTS, degree=SPLINE_DEGREE, include_bias=False
    )
    gam_mix = Pipeline(steps=[
        ("ct", ColumnTransformer([("spl", spline_mix, gam_mix_feats)], remainder="drop")),
        ("ridge", Ridge(alpha=RIDGE_ALPHA))
    ])

    # -------------------------
    # GroupKFold OOF evaluation
    # -------------------------
    n_splits = min(CV_SPLITS, groups.nunique())
    if n_splits < 2:
        raise ValueError(f"Need at least 2 unique groups for GroupKFold, got {groups.nunique()}")

    gkf = GroupKFold(n_splits=n_splits)

    oof = {
        "EMT": np.zeros(len(df), dtype=float),
        "EMT+Boost_param": np.zeros(len(df), dtype=float),
        "EMT+GAM_param": np.zeros(len(df), dtype=float),
        "EMT+Boost_mix": np.zeros(len(df), dtype=float),
        "EMT+GAM_mix": np.zeros(len(df), dtype=float),
    }

    R_emt_np = R_emt.to_numpy(dtype=float)

    for tr, te in gkf.split(X_param, y_resid, groups=groups):
        # param-only residual
        boost_param.fit(X_param.iloc[tr], y_resid.iloc[tr])
        resid_bparam = boost_param.predict(X_param.iloc[te])

        gam_param.fit(X_param.iloc[tr], y_resid.iloc[tr])
        resid_gparam = gam_param.predict(X_param.iloc[te])

        # mixing residual
        boost_mix.fit(X_mix.iloc[tr], y_resid.iloc[tr])
        resid_bmix = boost_mix.predict(X_mix.iloc[te])

        gam_mix.fit(X_gam_mix.iloc[tr], y_resid.iloc[tr])
        resid_gmix = gam_mix.predict(X_gam_mix.iloc[te])

        oof["EMT+Boost_param"][te] = R_emt_np[te] + resid_bparam
        oof["EMT+GAM_param"][te] = R_emt_np[te] + resid_gparam
        oof["EMT+Boost_mix"][te] = R_emt_np[te] + resid_bmix
        oof["EMT+GAM_mix"][te] = R_emt_np[te] + resid_gmix

    # EMT baseline prediction
    oof["EMT"] = R_emt_np

    # -------------------------
    # Two-mode metrics table (tau sweep supported)
    # -------------------------
    metrics_rows = []
    model_order = ["EMT", "EMT+Boost_param", "EMT+GAM_param", "EMT+Boost_mix", "EMT+GAM_mix"]
    for tau in taus:
        for name in model_order:
            m = metrics_two_modes(y_true_R.to_numpy(), oof[name], tau=float(tau))
            metrics_rows.append({"tau": float(tau), "model": name, **m})

    metrics_vs_tau = pd.DataFrame(metrics_rows)
    metrics_vs_tau_path = os.path.join(outdir, "metrics_param_vs_mixing_vs_tau.csv")
    metrics_vs_tau.to_csv(metrics_vs_tau_path, index=False)

    # Backward compatible single-tau output
    if len(taus) == 1:
        metrics_path = os.path.join(outdir, "metrics_param_vs_mixing.csv")
        metrics_vs_tau.drop(columns=["tau"]).to_csv(metrics_path, index=False)

    print("\n=== TWO beyond-EMT results (GroupKFold by scenario_id) ===")
    print("taus =", ", ".join([str(t) for t in taus]))
    print(metrics_vs_tau.to_string(index=False))
    print("\nSaved:", metrics_vs_tau_path)

    # Tau-sweep summary plots only (multi-tau + matplotlib)
    if HAS_MPL and len(taus) > 1:
        save_tau_summary_plot(metrics_vs_tau, outdir, ycol="r2_outbreak", filename="r2_outbreak_vs_tau.png")
        save_tau_summary_plot(metrics_vs_tau, outdir, ycol="mae_outbreak", filename="mae_outbreak_vs_tau.png")
        print("Saved:", os.path.join(outdir, "r2_outbreak_vs_tau.png"))
        print("Saved:", os.path.join(outdir, "mae_outbreak_vs_tau.png"))

    # Save OOF predictions (always)
    out = df.copy()
    out["R_emt_pred"] = R_emt_np
    out["R_pred_oof_EMT"] = oof["EMT"]
    out["R_pred_oof_boost_param"] = oof["EMT+Boost_param"]
    out["R_pred_oof_gam_param"] = oof["EMT+GAM_param"]
    out["R_pred_oof_boost_mix"] = oof["EMT+Boost_mix"]
    out["R_pred_oof_gam_mix"] = oof["EMT+GAM_mix"]
    out_csv = os.path.join(outdir, "runs_with_oof_predictions_param_vs_mixing.csv")
    out.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    # Fit final models on all data + export “theory” objects for GAMs (always)
    boost_param.fit(X_param, y_resid)
    boost_mix.fit(X_mix, y_resid)
    gam_param.fit(X_param, y_resid)
    gam_mix.fit(X_gam_mix, y_resid)

    joblib.dump(boost_param, os.path.join(outdir, "model_boost_param_residual.joblib"))
    joblib.dump(boost_mix, os.path.join(outdir, "model_boost_mix_residual.joblib"))
    joblib.dump(gam_param, os.path.join(outdir, "model_gam_param_residual.joblib"))
    joblib.dump(gam_mix, os.path.join(outdir, "model_gam_mix_residual.joblib"))
    print("Saved:", os.path.join(outdir, "model_boost_param_residual.joblib"))
    print("Saved:", os.path.join(outdir, "model_boost_mix_residual.joblib"))
    print("Saved:", os.path.join(outdir, "model_gam_param_residual.joblib"))
    print("Saved:", os.path.join(outdir, "model_gam_mix_residual.joblib"))

    export_gam_theory(gam_param, param_feats, os.path.join(outdir, "gam_param_theory.npz"))
    export_gam_theory(gam_mix, gam_mix_feats, os.path.join(outdir, "gam_mixing_theory.npz"))
    print("Saved:", os.path.join(outdir, "gam_param_theory.npz"))
    print("Saved:", os.path.join(outdir, "gam_mixing_theory.npz"))

    # Optional scatter/PDP plots (independent of tau count)
    if MAKE_PLOTS and HAS_MPL:
        yt = y_true_R.to_numpy(dtype=float)
        scatter_plot(yt, oof["EMT"], os.path.join(outdir, "scatter_EMT.png"), "Observed vs EMT")
        scatter_plot(yt, oof["EMT+GAM_param"], os.path.join(outdir, "scatter_GAM_param.png"), "Observed vs EMT+GAM (param-only)")
        scatter_plot(yt, oof["EMT+GAM_mix"], os.path.join(outdir, "scatter_GAM_mix.png"), "Observed vs EMT+GAM (mixing)")
        scatter_plot(yt, oof["EMT+Boost_param"], os.path.join(outdir, "scatter_Boost_param.png"), "Observed vs EMT+Boost (param-only)")
        scatter_plot(yt, oof["EMT+Boost_mix"], os.path.join(outdir, "scatter_Boost_mix.png"), "Observed vs EMT+Boost (mixing)")

        # Optional PDPs for GAMs, if helper exists
        try:
            Xref_param = X_param.sample(min(2000, len(X_param)), random_state=0).reset_index(drop=True)
            save_gam_pdps(gam_param, Xref_param, param_feats, outdir)

            Xref_mix = X_gam_mix.sample(min(2000, len(X_gam_mix)), random_state=0).reset_index(drop=True)
            save_gam_pdps(gam_mix, Xref_mix, gam_mix_feats, outdir)
        except Exception as e:
            print(f"[WARN] Skipped GAM PDP plots: {e}")

        print("Saved plots in:", outdir)
    elif MAKE_PLOTS and not HAS_MPL:
        print("matplotlib not found; skipped plots. (Install: python -m pip install matplotlib)")

    print("\nDONE. Output folder:", outdir)


if __name__ == "__main__":
    main()
