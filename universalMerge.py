import os, glob, ast
import numpy as np
import pandas as pd

# -----------------------
# Global parameters (edit if needed)
# -----------------------
GLOBAL = dict(
    v_fr=0.025, v_fb=0.025, v_fg=0.025,
    F=0.1, Frb=0.1, Frg=0.1, Fbg=0.1,
    v_rr=0.3, v_bb=0.3, v_gg=0.3,
    gamma=0.05,
    beta1=0.01, beta2=0.01, beta3=0.01,
    pb=1/3,
    N=4000, n=1000,
    runs=100, m=1000, N1=1000,
    denom_pr=30,
)

# -----------------------
# Scenario overrides (fill/update from your scenario.docx)
# Keys = scenario_id parsed from filename:
#   recover_<TAG>_<k>            -> scenario_id = TAG
#   recover_<TAG>_<variant>_<k>  -> scenario_id = TAG_<variant>
# Example: recover_July.14_2_19 -> scenario_id = "July.14_2"
# -----------------------
OVERRIDES = {
    "July.14":   dict(Frb=0.4),
    "July.14_1": dict(Frb=0.6),
    "July.14_2": dict(Frb=0.9),

    "July.17":   dict(Fbg=0.9),

    "July.18":   dict(v_fr=0.0025, Fbg=0.9),
    "July.18_1": dict(v_fr=0.0125, Fbg=0.9),

    "Aug.1":     dict(Fbg=0.6),
    "Aug.1_1":   dict(v_fr=0.0125),

    "Aug.2":     dict(v_fb=0.0125),
    "Aug.2_1":   dict(v_fg=0.0125),
    "Aug.2_2":   dict(v_rr=0.5),

    "Aug.3":     dict(v_gg=0.5),
    "Aug.8":     dict(v_bb=0.5),

    "Oct.4_1":   dict(Frg=0.6),
    "Oct.5":     dict(Frg=0.9),
}

INCLUDE = set(OVERRIDES.keys())   # only keep scenario_id's you listed in OVERRIDES


# -----------------------
# Outputs
# -----------------------
OUT_RUNS = "all_runs_merged.csv"
OUT_RISK = "all_risk_summary.csv"

# outbreak threshold for risk summary
TAU = 0.05

# -----------------------
# Helpers
# -----------------------
def strip_ext(name: str) -> str:
    return name[:-4] if name.lower().endswith(".csv") else name

def parse_scenario_and_k(recover_filename: str):
    """
    Accepts:
      recover_July.14_19
      recover_July.14_2_19
      recover_Aug.2_1_7
    Returns (scenario_id, k)
    """
    base = strip_ext(os.path.basename(recover_filename))
    if not base.startswith("recover_"):
        return None

    parts = base.split("_")
    if len(parts) < 3:
        return None

    tag = parts[1]
    try:
        k = int(parts[-1])
    except ValueError:
        return None

    variant = None if len(parts) == 3 else "_".join(parts[2:-1])
    scenario_id = tag if variant is None else f"{tag}_{variant}"
    return scenario_id, k

def parse_list_cell(x):
    return ast.literal_eval(str(x).strip())

def load_recover_long(path, scenario_id, k):
    df = pd.read_csv(path)
    if "recover_list" not in df.columns:
        raise ValueError(f"{path}: missing 'recover_list'")
    vals = df["recover_list"].astype(float).tolist()
    return pd.DataFrame({
        "scenario_id": scenario_id,
        "k": k,
        "run_id": np.arange(len(vals)),
        "R_final": vals,
    })

def load_infection_long(path, scenario_id, k):
    inf = pd.read_csv(path)
    if "interaction" not in inf.columns:
        raise ValueError(f"{path}: missing 'interaction'")

    lists = [parse_list_cell(v) for v in inf["interaction"].tolist()]
    if len(lists) < 10:
        raise ValueError(f"{path}: expected 10 rows (RR..GG + total_trans), got {len(lists)}")

    RR, RB, RG, BR, BB, BG, GR, GB, GG, total_trans = lists[:10]
    K = len(total_trans)

    return pd.DataFrame({
        "scenario_id": scenario_id,
        "k": k,
        "run_id": np.arange(K),
        "RR": RR, "RB": RB, "RG": RG,
        "BR": BR, "BB": BB, "BG": BG,
        "GR": GR, "GB": GB, "GG": GG,
        "total_trans": total_trans,
    })

def attach_params(df):
    df = df.copy()

    # initial infection metadata (your request)
    df["seed_species"] = "R"     # initial infected species is always Red
    df["seed_rule"] = "red_seed" # optional short label for your seeding method

    # k -> pr, and fixed pb
    df["pr"] = df["k"] / GLOBAL["denom_pr"]
    df["pb"] = GLOBAL["pb"]

    # defaults
    for key, val in GLOBAL.items():
        if key in ("denom_pr",):
            continue
        df[key] = val

    # per-scenario overrides
    for sid, ov in OVERRIDES.items():
        mask = (df["scenario_id"] == sid)
        if mask.any():
            for p, v in ov.items():
                df.loc[mask, p] = v

    return df

def add_features(df):
    eps = 1e-12
    off_diag = df["RB"] + df["RG"] + df["BR"] + df["BG"] + df["GR"] + df["GB"]
    within   = df["RR"] + df["BB"] + df["GG"]

    df["cross_trans_frac"] = (off_diag / (off_diag + within + eps)).astype(float)

    inter = df[["RR","RB","RG","BR","BB","BG","GR","GB","GG"]].to_numpy(dtype=float)
    p = inter / (inter.sum(axis=1, keepdims=True) + eps)
    df["interaction_entropy"] = (-p * np.log(p + eps)).sum(axis=1)
    return df

def risk_summary(df, tau=0.05):
    d = df.copy()
    d["outbreak"] = (d["R_final"] >= tau).astype(int)

    out = (
        d.groupby(["scenario_id","seed_species","k","pr","pb"])
         .agg(
            K=("R_final","size"),
            p_outbreak=("outbreak","mean"),
            R_mean=("R_final","mean"),
            R_q10=("R_final", lambda x: float(np.quantile(x, 0.10))),
            R_q50=("R_final", lambda x: float(np.quantile(x, 0.50))),
            R_q90=("R_final", lambda x: float(np.quantile(x, 0.90))),
            cross_trans_frac=("cross_trans_frac","mean"),
            interaction_entropy=("interaction_entropy","mean"),
            total_trans_mean=("total_trans","mean"),
         )
         .reset_index()
         .sort_values(["scenario_id","k"])
         .reset_index(drop=True)
    )
    return out

# -----------------------
# Main: merge everything
# -----------------------
recover_files = sorted(glob.glob("recover_*"))
merged_chunks = []
skipped = []

for rec in recover_files:
    parsed = parse_scenario_and_k(rec)
    if parsed is None:
        continue

    scenario_id, k = parsed
    if scenario_id not in INCLUDE:
        continue

    # Matching infection file = same suffix, per your guarantee
    inf = rec.replace("recover_", "Infection_", 1)
    if not os.path.exists(inf):
        if os.path.exists(inf + ".csv"):
            inf = inf + ".csv"
        elif rec.lower().endswith(".csv") and os.path.exists(inf[:-4]):
            inf = inf[:-4]
        else:
            skipped.append(f"{os.path.basename(rec)} (missing {os.path.basename(inf)})")
            continue

    try:
        rec_long = load_recover_long(rec, scenario_id, k)
        inf_long = load_infection_long(inf, scenario_id, k)
        m = rec_long.merge(inf_long, on=["scenario_id","k","run_id"], how="inner")
        if m.empty:
            skipped.append(f"{os.path.basename(rec)} (merge empty; run count mismatch?)")
            continue
        merged_chunks.append(m)
    except Exception as e:
        skipped.append(f"{os.path.basename(rec)} (error: {e})")

if skipped:
    print("\n[WARN] Skipped:")
    for s in skipped[:60]:
        print(" -", s)
    if len(skipped) > 60:
        print(" ...")

if not merged_chunks:
    raise SystemExit("No merged data produced. Check filenames and folder.")

df = pd.concat(merged_chunks, ignore_index=True)
df = attach_params(df)
df = add_features(df)

df.to_csv(OUT_RUNS, index=False)
print(f"\nSaved {OUT_RUNS} rows={len(df)}")

summ = risk_summary(df, tau=TAU)
summ.to_csv(OUT_RISK, index=False)
print(f"Saved {OUT_RISK} rows={len(summ)}")

print("\nRows per scenario_id (sanity):")
print(df.groupby("scenario_id").size().to_string())
