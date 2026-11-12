import numpy as np
from scipy.interpolate import BSpline

NPZ_PATH = r"C:\Users\marcu\PycharmProjects\pythonProject\beyond_emt_results_param_vs_mixing\gam_mixing_theory.npz"
# Or use gam_param_theory.npz:
# NPZ_PATH = r"C:\Users\marcu\PycharmProjects\pythonProject\beyond_emt_results_param_vs_mixing\gam_param_theory.npz"

def load_theory(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    feats = list(z["feature_names"])
    degree = int(z["degree"])
    include_bias = bool(z["include_bias"])
    knots = z["knots_"]          # object array: one knot vector per feature (in your export)
    coef = z["coef_"].astype(float)
    intercept = float(z["intercept_"])
    nspf = int(z["n_splines_per_feature"])
    return feats, degree, include_bias, knots, coef, intercept, nspf

def make_feature_splines(feats, degree, knots, coef, nspf):
    """
    Returns dict feat -> list[BSpline basis] and also combined f_j(x) via coefficients.
    Note: This constructs basis splines with unit coefficients for each basis index.
    """
    splines = {}
    idx = 0
    for j, feat in enumerate(feats):
        t = np.array(knots[j], dtype=float)

        # Build basis splines B_{j,m}(x) of degree 'degree' on knot vector t
        # Each basis has coefficient vector c with a single 1 at position m.
        # Number of basis functions for a knot vector t and degree k is: len(t) - k - 1
        n_basis = len(t) - degree - 1

        # In sklearn SplineTransformer, n_basis should match nspf (per feature).
        # We'll align by taking min if any mismatch.
        n_use = min(n_basis, nspf)
        basis_list = []
        for m in range(n_use):
            c = np.zeros(n_basis)
            c[m] = 1.0
            basis_list.append(BSpline(t, c, degree, extrapolate=True))
        splines[feat] = basis_list

        idx += nspf
    return splines

def print_symbolic(feats, degree, knots, coef, intercept, nspf, max_terms=8):
    print("\n=== GAM-spline residual correction law ===")
    print("Form:  Δ(x) = b0 + Σ_j Σ_m a_{j,m} B_{j,m}(x_j)")
    print(f"b0 (intercept) = {intercept:.6g}")
    print(f"degree = {degree}, n_splines_per_feature = {nspf}")

    print("\nEquivalent additive form: Δ(x) = b0 + Σ_j f_j(x_j),  f_j(x)=Σ_m a_{j,m}B_{j,m}(x)\n")

    # Coeffs are concatenated feature-by-feature in the same order as feature_names
    idx = 0
    for feat_i, feat in enumerate(feats):
        a = coef[idx: idx + nspf]
        t = np.array(knots[feat_i], dtype=float)

        print(f"--- Feature: {feat} ---")
        print(f"knots t_{feat} = {np.array2string(t, precision=6, separator=',')}")
        print(f"coeffs a_{feat}[m] (first {min(max_terms, len(a))} of {len(a)}):")
        print(np.array2string(a[:max_terms], precision=6, separator=', '))

        # “symbolic” line
        show = min(max_terms, len(a))
        terms = " + ".join([f"({a[m]:.3g})*B_{feat},{m}(x_{feat})" for m in range(show)])
        if len(a) > show:
            terms += " + ..."
        print(f"f_{feat}(x_{feat}) = {terms}\n")

        idx += nspf

def build_predictor(npz_path):
    feats, degree, include_bias, knots, coef, intercept, nspf = load_theory(npz_path)
    splines = make_feature_splines(feats, degree, knots, coef, nspf)

    # Pre-split coefficients per feature
    coef_map = {}
    idx = 0
    for feat in feats:
        coef_map[feat] = coef[idx: idx + nspf].copy()
        idx += nspf

    def delta(x_dict):
        # x_dict: {feat: float}
        val = intercept
        for feat in feats:
            x = float(x_dict[feat])
            a = coef_map[feat]
            basis_list = splines[feat]
            # Evaluate each basis and sum
            s = 0.0
            for m, B in enumerate(basis_list):
                s += a[m] * float(B(x))
            val += s
        return float(val)

    return feats, delta

if __name__ == "__main__":
    feats, degree, include_bias, knots, coef, intercept, nspf = load_theory(NPZ_PATH)
    print_symbolic(feats, degree, knots, coef, intercept, nspf)

    # Optional: quick numerical test evaluation (you must provide values for all features)
    # Example skeleton:
    # feats, delta_fn = build_predictor(NPZ_PATH)
    # x0 = {f: 0.0 for f in feats}  # REPLACE with realistic feature values
    # print("Δ(x0) =", delta_fn(x0))