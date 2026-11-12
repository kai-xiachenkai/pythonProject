import os, re, glob, ast
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
DENOM = 30
PB_FIXED = 1/3
TAU = 0.05  # outbreak threshold on R_final; also try 0.01 later

REC_GLOB = "recover_July.14_*"
OUT_MERGED = "July14_merged_runs.csv"
OUT_RISK   = "July14_risk_summary.csv"

# Map "batch" in the TWO-number filenames to Frb
FRB_BY_BATCH = {1: 0.6, 2: 0.9}
FRB_SINGLE_NUMBER = 0.4   # recover_July.14_<k>

# ---------------- HELPERS ----------------
def parse_k_and_frb_from_recover(path: str):
    """
    Accepts:
      recover_July.14_<k>            -> Frb = 0.4
      recover_July.14_<batch>_<k>    -> Frb = FRB_BY_BATCH[batch]
    Optional .csv extension allowed.
    Returns dict with keys: k, Frb, pr, batch (batch=None for single-number)
    """
    base = os.path.basename(path)

    # Two-number form: recover_July.14_<batch>_<k>
    m2 = re.match(r"^recover_July\.14_(\d+)_(\d+)(?:\.csv)?$", base)
    if m2:
        batch = int(m2.group(1))
        k = int(m2.group(2))
        if batch not in FRB_BY_BATCH:
            raise ValueError(f"Unknown batch={batch} in filename {base}. Add it to FRB_BY_BATCH.")
        frb = FRB_BY_BATCH[batch]
        return {"batch": batch, "k": k, "Frb": frb, "pr": k / DENOM}

    # One-number form: recover_July.14_<k>
    m1 = re.match(r"^recover_July\.14_(\d+)(?:\.csv)?$", base)
    if m1:
        k = int(m1.group(1))
        return {"batch": None, "k": k, "Frb": FRB_SINGLE_NUMBER, "pr": k / DENOM}

    return None

def parse_list_cell(x):
    # Infection CSV has cells like "[0, 0, 1, ...]"
    return ast.literal_eval(str(x).strip())

def load_recover_long(recover_path, meta):
    df = pd.read_csv(recover_path)
    if "recover_list" not in df.columns:
        raise ValueError(f"{recover_path}: missing 'recover_list' column. Columns={df.columns.tolist()}")

    vals = df["recover_list"].astype(float).tolist()
    out = pd.DataFrame({
        "Frb": meta["Frb"],
        "batch": meta["batch"] if meta["batch"] is not None else 0,  # 0 for the single-number group
        "k": meta["k"],
        "pr": meta["pr"],
        "pb": PB_FIXED,
        "run_id": np.arange(len(vals)),
        "R_final": vals,
    })
    return out

def load_infection_long(infection_path, meta):
    inf = pd.read_csv(infection_path)
    if "interaction" not in inf.columns:
        raise ValueError(f"{infection_path}: missing 'interaction' column. Columns={inf.columns.tolist()}")

    lists = [parse_list_cell(v) for v in inf["interaction"].tolist()]
    if len(lists) < 10:
        raise ValueError(f"{infection_path}: expected 10 rows (RR..GG + total_trans). Got {len(lists)}")

    RR, RB, RG, BR, BB, BG, GR, GB, GG, total_trans = lists[:10]
    K = len(total_trans)

    out = pd.DataFrame({
        "Frb": meta["Frb"],
        "batch": meta["batch"] if meta["batch"] is not None else 0,
        "k": meta["k"],
        "pr": meta["pr"],
        "pb": PB_FIXED,
        "run_id": np.arange(K),
        "RR": RR, "RB": RB, "RG": RG,
        "BR": BR, "BB": BB, "BG": BG,
        "GR": GR, "GB": GB, "GG": GG,
        "total_trans": total_trans,
    })
    return out

def add_features(df):
    eps = 1e-12
    off_diag = df["RB"] + df["RG"] + df["BR"] + df["BG"] + df["GR"] + df["GB"]
    within   = df["RR"] + df["BB"] + df["GG"]

    df["cross_trans_frac"]  = (off_diag / (off_diag + within + eps)).astype(float)
    df["within_trans_frac"] = (within  / (off_diag + within + eps)).astype(float)

    inter = df[["RR","RB","RG","BR","BB","BG","GR","GB","GG"]].to_numpy(dtype=float)
    p = inter / (inter.sum(axis=1, keepdims=True) + eps)
    df["interaction_entropy"] = (-p * np.log(p + eps)).sum(axis=1)
    return df

def risk_summary(df):
    df = df.copy()
    df["outbreak"] = (df["R_final"] >= TAU).astype(int)

    out = (
        df.groupby(["Frb","batch","k","pr","pb"])
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
          .sort_values(["Frb","k"])
          .reset_index(drop=True)
    )
    return out

# ---------------- MAIN ----------------
all_merged = []
skipped = []

for rec in sorted(glob.glob(REC_GLOB)):
    meta = parse_k_and_frb_from_recover(rec)
    if meta is None:
        skipped.append(os.path.basename(rec))
        continue

    # Matching infection file name by prefix swap
    inf = rec.replace("recover_", "Infection_", 1)

    # Handle possible .csv mismatches
    if not os.path.exists(inf):
        if os.path.exists(inf + ".csv"):
            inf = inf + ".csv"
        elif rec.endswith(".csv") and os.path.exists(inf[:-4]):
            inf = inf[:-4]
        else:
            skipped.append(f"{os.path.basename(rec)} (missing {os.path.basename(inf)})")
            continue

    rec_long = load_recover_long(rec, meta)
    inf_long = load_infection_long(inf, meta)

    merged = rec_long.merge(inf_long, on=["Frb","batch","k","pr","pb","run_id"], how="inner")
    if merged.empty:
        skipped.append(f"{os.path.basename(rec)} (merge empty: run count mismatch?)")
        continue

    all_merged.append(merged)

if skipped:
    print("\n[WARN] Skipped files:")
    for s in skipped[:80]:
        print(" -", s)
    if len(skipped) > 80:
        print(" - ...")

if not all_merged:
    raise SystemExit("No data merged. Check filenames and that Infection files exist for each recover file.")

df = pd.concat(all_merged, ignore_index=True)
df = add_features(df)
df.to_csv(OUT_MERGED, index=False)
print(f"\nSaved {OUT_MERGED} rows={len(df)}")

print("\nRows per (Frb,k): (expect ~100 each, times any extra file multiplicity)")
print(df.groupby(["Frb","k"]).size().to_string())

summ = risk_summary(df)
summ.to_csv(OUT_RISK, index=False)
print(f"\nSaved {OUT_RISK} rows={len(summ)}")
print("\nPreview risk summary:")
print(summ.head(12).to_string(index=False))
