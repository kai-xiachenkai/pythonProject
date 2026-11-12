import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# HARD-CODED PATHS
# =========================
BASE_DIR = r"C:\Users\marcu\PycharmProjects\pythonProject"
DATA_CSV = os.path.join(BASE_DIR, "all_runs_merged.csv")
second_DIR = r"C:\Users\marcu\PycharmProjects\pythonProject\beyond_emt_results_param_vs_mixing"
EMT_CACHE = os.path.join(second_DIR, "emt_cache.csv")

# All outputs go here (upload these PNGs to Overleaf later)
OUT_DIR = os.path.join(BASE_DIR, "figs_for_overleaf")
os.makedirs(OUT_DIR, exist_ok=True)

# Heatmap stability: require at least this many points per bin
MIN_BIN_COUNT = 10

# Binning resolution for heatmap
N_BINS = 35


def main():
    print("Loading:", DATA_CSV)
    df = pd.read_csv(DATA_CSV)

    print("Loading:", EMT_CACHE)
    cache = pd.read_csv(EMT_CACHE)

    # Normalize types for merge keys
    df["scenario_id"] = df["scenario_id"].astype(str)
    cache["scenario_id"] = cache["scenario_id"].astype(str)

    # Merge EMT baseline
    df = df.merge(cache[["scenario_id", "k", "R_emt_pred"]], on=["scenario_id", "k"], how="left")
    missing = int(df["R_emt_pred"].isna().sum())
    if missing > 0:
        raise ValueError(f"Missing R_emt_pred after merge for {missing} rows. Check emt_cache.csv vs data keys.")

    # Derived features
    df["A"] = np.log1p(df["total_trans"].astype(float))  # A = log(1+T)
    df["Delta"] = df["R_final"].astype(float) - df["R_emt_pred"].astype(float)

    # ----------------------------
    # 1) Baseline scatter: R_inf vs R_emt
    # ----------------------------
    plt.figure(figsize=(7.0, 6.0))
    plt.scatter(df["R_emt_pred"], df["R_final"], alpha=0.08, s=8)

    mn = float(min(df["R_emt_pred"].min(), df["R_final"].min()))
    mx = float(max(df["R_emt_pred"].max(), df["R_final"].max()))
    plt.plot([mn, mx], [mn, mx], linewidth=1)

    plt.xlabel(r"$R_{\mathrm{emt}}$")
    plt.ylabel(r"$R_{\infty}$")
    plt.title(r"Baseline: $R_{\infty}$ vs $R_{\mathrm{emt}}$ (all runs)")
    plt.tight_layout()

    p1 = os.path.join(OUT_DIR, "collapse_baseline_Rinf_vs_Remt.png")
    plt.savefig(p1, dpi=250)
    plt.close()
    print("Saved:", p1)

    # ----------------------------
    # 2) Collapse scatter: Delta vs A
    # ----------------------------
    plt.figure(figsize=(7.0, 6.0))
    plt.scatter(df["A"], df["Delta"], alpha=0.08, s=8)
    plt.axhline(0, linewidth=1)

    plt.xlabel(r"$A=\log(1+T)$")
    plt.ylabel(r"$\Delta = R_{\infty}-R_{\mathrm{emt}}$")
    plt.title(r"Collapse view: $\Delta$ vs $A=\log(1+T)$ (all runs)")
    plt.tight_layout()

    p2 = os.path.join(OUT_DIR, "collapse_Delta_vs_A.png")
    plt.savefig(p2, dpi=250)
    plt.close()
    print("Saved:", p2)

    # ----------------------------
    # 3) Heatmap: mean Delta on (R_emt, A)
    # ----------------------------
    x = df["R_emt_pred"].to_numpy(dtype=float)
    y = df["A"].to_numpy(dtype=float)
    z = df["Delta"].to_numpy(dtype=float)

    x_bins = np.linspace(np.nanmin(x), np.nanmax(x), N_BINS)
    y_bins = np.linspace(np.nanmin(y), np.nanmax(y), N_BINS)

    H_sum = np.zeros((len(y_bins) - 1, len(x_bins) - 1), dtype=float)
    H_cnt = np.zeros((len(y_bins) - 1, len(x_bins) - 1), dtype=float)

    xi = np.clip(np.digitize(x, x_bins) - 1, 0, len(x_bins) - 2)
    yi = np.clip(np.digitize(y, y_bins) - 1, 0, len(y_bins) - 2)

    for i in range(len(z)):
        H_sum[yi[i], xi[i]] += z[i]
        H_cnt[yi[i], xi[i]] += 1.0

    H_mean = np.full_like(H_sum, np.nan)
    mask = H_cnt >= MIN_BIN_COUNT
    H_mean[mask] = H_sum[mask] / H_cnt[mask]

    plt.figure(figsize=(7.5, 6.0))
    plt.imshow(
        H_mean,
        origin="lower",
        aspect="auto",
        extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
    )
    plt.colorbar(label=r"Mean $\Delta$")
    plt.xlabel(r"$R_{\mathrm{emt}}$")
    plt.ylabel(r"$A=\log(1+T)$")
    plt.title(r"Mean residual surface: $\mathbb{E}[\Delta \mid R_{\mathrm{emt}},A]$")
    plt.tight_layout()

    p3 = os.path.join(OUT_DIR, "collapse_heatmap_mean_Delta_Remt_A.png")
    plt.savefig(p3, dpi=250)
    plt.close()
    print("Saved:", p3)

    print("\nDONE. Upload these to Overleaf from:")
    print(" ", OUT_DIR)


if __name__ == "__main__":
    main()