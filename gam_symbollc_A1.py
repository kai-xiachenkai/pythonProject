import argparse
import os
import numpy as np

# PyCharm-friendly default (relative to this script file)
DEFAULT_NPZ = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "beyond_emt_ablation",
    "gam_A1_theory.npz",
)

DEFAULT_SHOW_COEFFS = 8


def main():
    ap = argparse.ArgumentParser()
    # no longer required=True
    ap.add_argument("--npz", default=DEFAULT_NPZ, help="Path to gam_A1_theory.npz")
    ap.add_argument("--show_coeffs", type=int, default=DEFAULT_SHOW_COEFFS,
                    help="How many coeffs to print per feature (preview)")
    args = ap.parse_args()

    if not os.path.exists(args.npz):
        raise FileNotFoundError(f"NPZ file not found: {args.npz}")

    z = np.load(args.npz, allow_pickle=True)
    feats = list(z["feature_names"])
    degree = int(z["degree"])
    nspf = int(z["n_splines_per_feature"])
    knots = list(z["knots_"])
    coef = z["coef_"].astype(float).reshape(-1)
    intercept = float(z["intercept_"])

    print("=== GAM A1 symbolic form (residual closure) ===")
    print("Residual: Δ = R_final - R_emt_pred")
    print(f"Δ(x) = intercept + Σ_j f_j(x_j),  intercept = {intercept:.6g}")
    print(f"degree = {degree}, n_splines_per_feature = {nspf}")
    print()

    idx = 0
    for j, feat in enumerate(feats):
        a = coef[idx:idx+nspf]
        idx += nspf
        t = np.array(knots[j], dtype=float)

        print(f"--- Feature: {feat} ---")
        print("knots t =", np.array2string(t, precision=6, separator=","))
        k = min(len(a), args.show_coeffs)
        print(f"coeffs a_{feat}[m] (first {k} of {len(a)}):")
        print(np.array2string(a[:k], precision=6, separator=", "))
        terms = [f"({a[m]:.6g})*B_{feat},{m}({feat})" for m in range(k)]
        s = " + ".join(terms) + (" + ..." if len(a) > k else "")
        print(f"f_{feat}({feat}) = {s}")
        print()

    print("Final-size prediction form:")
    print("R_pred = R_emt_pred + Δ(x)")
    print("with x = [R_emt_pred, log_total_trans]")


if __name__ == "__main__":
    main()