import os
import sys

# =========================
# HARD-CODED SETTINGS
# =========================
INPUT_CSV = "all_runs_merged.csv"       # put this CSV in the SAME folder as this .py
OUTDIR = "beyond_emt_results"           # output folder (created next to this .py)
BETA_COL = "beta1"                      # your CSV has beta1,beta2,beta3; use beta1
TAU = 0.05                              # outbreak threshold for "outbreak-only" evaluation
CV_SPLITS = 5                           # GroupKFold splits by scenario_id
SPLINE_KNOTS = 8                        # GAM-like splines knots
SPLINE_DEGREE = 3                       # GAM-like splines degree
RIDGE_ALPHA = 1.0                       # GAM ridge regularization


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
    """Save a simple summary plot of ycol vs tau for each model."""
    if not HAS_MPL:
        return
    if "tau" not in metrics_vs_tau.columns:
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


# =========================
# Dependency guard (prints the exact pip command)
# =========================
def _require(pkg_name, import_name=None):
    try:
        __import__(import_name or pkg_name)
    except Exception as e:
        print("\n[IMPORT ERROR]", e)
        print("\nYou're running this script with a Python interpreter that doesn't have required packages.")
        print("In PyCharm: Settings -> Project -> Python Interpreter, choose the interpreter you want.")
        print("\nThen install deps in THAT interpreter terminal with:")
        print("  python -m pip install numpy pandas scipy scikit-learn joblib matplotlib")
        print("\n(Windows alternative if needed):")
        print("  py -m pip install numpy pandas scipy scikit-learn joblib matplotlib")
        sys.exit(1)


_require("numpy")
_require("pandas")
_require("scipy")
_require("sklearn", "sklearn")
_require("joblib")
# matplotlib is optional; we’ll try it and skip plots if missing
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

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
# Your EMT reduced ODE: state = S(t)
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
# Helpers
# =========================
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
    need = ["total_trans", "RR","RB","RG","BR","BB","BG","GR","GB","GG"]
    ensure_cols(df, need)

    df = df.copy()
    T = df["total_trans"].astype(float)
    T_safe = T.replace(0, np.nan)

    inter = ["RR","RB","RG","BR","BB","BG","GR","GB","GG"]
    for c in inter:
        df[f"{c}_share"] = (df[c].astype(float) / T_safe).fillna(0.0)

    P = df[[f"{c}_share" for c in inter]].to_numpy(dtype=float)
    df["interaction_hhi"] = (P**2).sum(axis=1)
    df["log_total_trans"] = np.log1p(T.astype(float))

    return df


def compute_emt_baseline_cached(df, cache_csv, beta_col=BETA_COL):
    need = ["scenario_id","k","pr","pb","n",
            "v_fr","v_fb","v_fg","v_rr","v_bb","v_gg","gamma", beta_col]
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

    # load existing cache if present
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
            return float(R_cnt) / float(r.n)  # fraction

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
    Exports an explicit additive-spline “theory object” for the residual:
      Δ = b0 + Σ_j Σ_k a_{j,k} B_{j,k}(x_j)

    Compatible with sklearn versions that expose either `knots_` or `bsplines_`.
    """
    ct = gam_model.named_steps["ct"]
    ridge = gam_model.named_steps["ridge"]
    spl = ct.named_transformers_["spl"]

    feature_names = np.array(gam_feats, dtype=object)
    degree = int(getattr(spl, "degree", 3))
    include_bias = bool(getattr(spl, "include_bias", False))

    # --- Extract knots robustly ---
    knots = None
    if hasattr(spl, "knots_"):
        knots = spl.knots_.copy()
    elif hasattr(spl, "bsplines_"):
        # bsplines_ contains SciPy BSpline objects; each has `.t` as knot vector
        bs = spl.bsplines_
        knot_list = []
        for item in bs:
            # Depending on sklearn version, item can be a BSpline or a list of BSplines
            if isinstance(item, list):
                knot_list.append(np.array(item[0].t))
            else:
                knot_list.append(np.array(item.t))
        knots = np.array(knot_list, dtype=object)
    else:
        # Worst-case fallback: store nothing but still export coefficients/intercept
        knots = np.array([], dtype=object)

    coef = ridge.coef_.copy()
    intercept = float(ridge.intercept_)

    n_features = len(gam_feats)
    # number of output spline basis columns
    try:
        n_out = int(spl.n_features_out_)
    except Exception:
        # compute from a dummy transform
        n_out = int(spl.transform(np.zeros((1, n_features))).shape[1])

    n_splines_per_feature = int(n_out // n_features) if n_features > 0 else 0

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



def save_gam_pdps(gam_model, X_ref, feature_cols, outdir):
    if not HAS_MPL:
        return
    # median-hold PDP per feature
    X0 = X_ref.copy()
    for c in feature_cols:
        X0[c] = np.median(X0[c].to_numpy(dtype=float))

    for feat in feature_cols:
        lo = float(np.quantile(X_ref[feat].to_numpy(dtype=float), 0.01))
        hi = float(np.quantile(X_ref[feat].to_numpy(dtype=float), 0.99))
        grid = np.linspace(lo, hi, 80)

        preds = []
        for g in grid:
            Xg = X0.copy()
            Xg[feat] = g
            preds.append(float(gam_model.predict(Xg)[0]))
        preds = np.array(preds)

        plt.figure()
        plt.plot(grid, preds)
        plt.xlabel(feat)
        plt.ylabel("Predicted residual Δ")
        plt.title(f"GAM-like term effect: {feat}")
        plt.savefig(os.path.join(outdir, f"gam_pdp_{feat}.png"), dpi=200, bbox_inches="tight")
        plt.close()


def scatter_plot(y_true, y_pred, outpath, title):
    if not HAS_MPL:
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
        print("Fix by either:")
        print("  1) Put all_runs_merged.csv in the same folder as this .py, or")
        print("  2) Set INPUT_CSV to the full absolute path.")
        sys.exit(1)

    df = pd.read_csv(input_path)

    # Validate your known header essentials
    ensure_cols(df, [
        "scenario_id", "k", "run_id", "R_final",
        "RR", "RB", "RG", "BR", "BB", "BG", "GR", "GB", "GG", "total_trans",
        "pr", "pb", "v_fr", "v_fb", "v_fg", "Frb", "Frg", "Fbg", "v_rr", "v_bb", "v_gg",
        "gamma", BETA_COL, "n",
        "cross_trans_frac", "interaction_entropy"
    ])

    # EMT baseline (cached per (scenario_id,k))
    emt_cache_csv = os.path.join(outdir, "emt_cache.csv")
    df = compute_emt_baseline_cached(df, cache_csv=emt_cache_csv, beta_col=BETA_COL)

    # Add extra mixing features (shares + hhi + log_total_trans)
    df = add_mixing_features(df)

    # Build targets and groups
    y_true_R = df["R_final"].astype(float)
    y_resid = df["residual_vs_emt"].astype(float)
    R_emt = df["R_emt_pred"].astype(float)
    groups = df["scenario_id"].astype(str)

    # Features
    gam_feats = ["R_emt_pred", "pr", "cross_trans_frac", "interaction_entropy", "log_total_trans", "interaction_hhi"]
    share_feats = [c for c in df.columns if c.endswith("_share")]
    boost_feats = gam_feats + sorted(share_feats)

    X_gam = df[gam_feats].replace([np.inf, -np.inf], np.nan)
    X_boost = df[boost_feats].replace([np.inf, -np.inf], np.nan)

    good = (~X_boost.isna().any(axis=1)) & (~y_resid.isna()) & (~y_true_R.isna()) & (~R_emt.isna())
    df = df.loc[good].reset_index(drop=True)
    X_gam = X_gam.loc[good].reset_index(drop=True)
    X_boost = X_boost.loc[good].reset_index(drop=True)
    y_true_R = y_true_R.loc[good].reset_index(drop=True)
    y_resid = y_resid.loc[good].reset_index(drop=True)
    R_emt = R_emt.loc[good].reset_index(drop=True)
    groups = groups.loc[good].reset_index(drop=True)

    # Models predict residual
    boost_model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=700,
        random_state=42,
    )

    spline = SplineTransformer(
        n_knots=SPLINE_KNOTS,
        degree=SPLINE_DEGREE,
        include_bias=False,
    )
    gam_model = Pipeline(steps=[
        ("ct", ColumnTransformer(
            transformers=[("spl", spline, gam_feats)],
            remainder="drop"
        )),
        ("ridge", Ridge(alpha=RIDGE_ALPHA))
    ])

    # GroupKFold OOF on residual
    n_splits = min(CV_SPLITS, groups.nunique())
    if n_splits < 2:
        raise ValueError(f"Need at least 2 unique groups for GroupKFold, got {groups.nunique()}")

    gkf = GroupKFold(n_splits=n_splits)

    oof_resid_boost = np.zeros(len(df), dtype=float)
    oof_resid_gam = np.zeros(len(df), dtype=float)

    for tr, te in gkf.split(X_boost, y_resid, groups=groups):
        boost_model.fit(X_boost.iloc[tr], y_resid.iloc[tr])
        oof_resid_boost[te] = boost_model.predict(X_boost.iloc[te])

        gam_model.fit(X_gam.iloc[tr], y_resid.iloc[tr])
        oof_resid_gam[te] = gam_model.predict(X_gam.iloc[te])

    R_pred_emt = R_emt.to_numpy(dtype=float)
    R_pred_boost = R_pred_emt + oof_resid_boost
    R_pred_gam = R_pred_emt + oof_resid_gam

    # ---- TWO RESULTS: all runs + outbreak-only ----
    # (Tau only affects the outbreak-only metrics; predictions are computed once.)
    metrics_rows = []
    for tau in taus:
        for name, pred in [
            ("EMT", R_pred_emt),
            ("EMT+Boosting", R_pred_boost),
            ("EMT+GAM_splines", R_pred_gam),
        ]:
            m = metrics_two_modes(y_true_R.to_numpy(), pred, tau=float(tau))
            metrics_rows.append({"tau": float(tau), "model": name, **m})

    metrics_vs_tau = pd.DataFrame(metrics_rows)
    metrics_vs_tau_path = os.path.join(outdir, "metrics_vs_tau.csv")
    metrics_vs_tau.to_csv(metrics_vs_tau_path, index=False)

    # Backward compatible single-tau output
    if len(taus) == 1:
        metrics_path = os.path.join(outdir, "metrics.csv")
        metrics_vs_tau.drop(columns=["tau"]).to_csv(metrics_path, index=False)

    print("\n=== TWO beyond-EMT results (GroupKFold by scenario_id) ===")
    print("taus =", ", ".join([str(t) for t in taus]))
    print(metrics_vs_tau.to_string(index=False))
    print("\nSaved:", metrics_vs_tau_path)

    # Tau-summary plots only if matplotlib and multiple taus
    if HAS_MPL and len(taus) > 1:
        save_tau_summary_plot(metrics_vs_tau, outdir, ycol="r2_outbreak", filename="r2_outbreak_vs_tau.png")
        save_tau_summary_plot(metrics_vs_tau, outdir, ycol="mae_outbreak", filename="mae_outbreak_vs_tau.png")
        print("Saved:", os.path.join(outdir, "r2_outbreak_vs_tau.png"))
        print("Saved:", os.path.join(outdir, "mae_outbreak_vs_tau.png"))

    # Save per-run predictions (should NOT depend on matplotlib / multi-tau)
    out = df.copy()
    out["R_emt_pred"] = R_pred_emt
    out["R_pred_oof_boost"] = R_pred_boost
    out["R_pred_oof_gam"] = R_pred_gam
    out["resid_pred_oof_boost"] = oof_resid_boost
    out["resid_pred_oof_gam"] = oof_resid_gam
    out_csv = os.path.join(outdir, "runs_with_oof_predictions.csv")
    out.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    # Fit final models on all data + save
    boost_model.fit(X_boost, y_resid)
    gam_model.fit(X_gam, y_resid)
    joblib.dump(boost_model, os.path.join(outdir, "model_boost_residual.joblib"))
    joblib.dump(gam_model, os.path.join(outdir, "model_gam_spline_residual.joblib"))
    print("Saved:", os.path.join(outdir, "model_boost_residual.joblib"))
    print("Saved:", os.path.join(outdir, "model_gam_spline_residual.joblib"))

    # Export explicit GAM “final prediction theory”
    export_gam_theory(
        gam_model=gam_model,
        gam_feats=gam_feats,
        outpath_npz=os.path.join(outdir, "gam_spline_theory.npz")
    )
    print("Saved:", os.path.join(outdir, "gam_spline_theory.npz"))

    # Optional plots
    if HAS_MPL:
        scatter_plot(y_true_R.to_numpy(), R_pred_emt, os.path.join(outdir, "scatter_emt.png"), "Observed vs EMT")
        scatter_plot(y_true_R.to_numpy(), R_pred_gam, os.path.join(outdir, "scatter_gam.png"), "Observed vs EMT+GAM (OOF)")
        scatter_plot(y_true_R.to_numpy(), R_pred_boost, os.path.join(outdir, "scatter_boost.png"), "Observed vs EMT+Boosting (OOF)")

        Xref = X_gam.sample(min(2000, len(X_gam)), random_state=0).reset_index(drop=True)
        save_gam_pdps(gam_model, Xref, gam_feats, outdir)
        print("Saved plots in:", outdir)
    else:
        print("matplotlib not found; skipped plots. (Install: python -m pip install matplotlib)")

    print("\nDONE. Output folder:", outdir)


if __name__ == "__main__":
    main()
