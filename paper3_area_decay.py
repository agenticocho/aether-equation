#!/usr/bin/env python3
"""paper3_area_decay.py — Compute A(K) area decay and fit γ exponent for each g-function.

For the generalized Aether iteration z_{n+1} = K · z · g(Im(z)) + c,
A(K) is the measure of the (BSS, δ) plane that remains convergent.

Method:
    Uses an adaptive viewport that shrinks with K — the stable region
    contracts as K grows, so the sampling window must track it.
    For each (g, K):
        1. Estimate R_eff ~ R/K as the scale of the convergent set
        2. Sample on [-R_eff, R_eff]^2 at high resolution
        3. Count convergent pixels → fractional area × viewport_area = A(K)
    Then fit A(K) ~ C · K^{-γ} via log-log regression.

Outputs:
    - area_decay_sweep.csv
    - gamma_fit_results.csv

Usage:
    python3 paper3_area_decay.py [--resolution 1000] [--iter 500]

License: MIT
"""

import argparse
import csv
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper3_sweep_render import G_FUNCTIONS, render_aether_generalized


# ── K values: finer log-spacing, focused on measurable range ─────────────────
K_SWEEP_SMALL = [2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500,
                 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000]

# Theoretical γ predictions
GAMMA_PREDICTED = {
    "power_alpha_0.5":  2.0 / (1.0 + 0.5),    # 4/3 ≈ 1.333
    "power_alpha_0.75": 2.0 / (1.0 + 0.75),   # 8/7 ≈ 1.143
    "power_alpha_1.0":  2.0 / (1.0 + 1.0),    # 1.0
    "power_alpha_2.0":  2.0 / (1.0 + 2.0),    # 2/3 ≈ 0.667
    "exp_neg_abs":      2.0,                    # Class A: g(0) = 1 > 0
    "tanh_abs":         1.0,                    # Class B: vanishing order α=1
    "sigmoid_abs":      2.0,                    # Class A: g(0) = 0.5 > 0
}

# Approximate initial convergent set radius for viewport scaling
# exp(Im(z)) at z~0 is ~1, so the set is roughly within |c| < R/(K*g(0)+1)
G_ZERO = {
    "power_alpha_0.5":  0.001,   # g(0) ≈ 0 (epsilon)
    "power_alpha_0.75": 0.001,
    "power_alpha_1.0":  0.001,
    "power_alpha_2.0":  0.001,
    "exp_neg_abs":      1.0,     # e^0 = 1
    "tanh_abs":         0.001,   # tanh(0) = 0
    "sigmoid_abs":      0.5,     # σ(0) = 0.5
}


def adaptive_viewport(K, g_name, R=50.0):
    """Compute an adaptive viewport that tracks the shrinking convergent set.

    For g(0) > 0 (Class A): set radius ~ R / K
    For g(0) ~ 0 (vanishing): the set survives longer, use wider viewport
    """
    g0 = G_ZERO.get(g_name, 0.5)
    if g0 > 0.01:
        # Class A: convergent set ~ R / (K * g0)
        scale = min(R / max(K * g0, 1), 2.0)
    else:
        # Vanishing: set is larger. |y|^α near 0 lets orbits survive.
        # Empirical: scale ~ R / K^(1/(1+α)) is a reasonable first guess
        scale = min(R / max(K ** 0.4, 1), 2.0)

    # Minimum viewport to avoid numerical noise
    scale = max(scale, 0.001)
    return (-scale, scale, -scale, scale)


def measure_area_adaptive(K, g_func, g_name, resolution, max_iter, escape_radius):
    """Compute A(K) using adaptive viewport."""
    bss_lo, bss_hi, delta_lo, delta_hi = adaptive_viewport(K, g_name, escape_radius)
    viewport_area = (bss_hi - bss_lo) * (delta_hi - delta_lo)

    escape, _ = render_aether_generalized(
        K, g_func, resolution, resolution, max_iter,
        escape_radius, (bss_lo, bss_hi), (delta_lo, delta_hi))

    convergent = np.sum(escape == max_iter)
    total = resolution * resolution
    return (convergent / total) * viewport_area, viewport_area


def fit_gamma(k_vals, a_vals):
    """Fit γ from A(K) data via log-log regression."""
    k_arr = np.array(k_vals, dtype=np.float64)
    a_arr = np.array(a_vals, dtype=np.float64)

    mask = a_arr > 0
    if mask.sum() < 3:
        return None, None, None, int(mask.sum())

    log_k = np.log(k_arr[mask])
    log_a = np.log(a_arr[mask])

    coeffs = np.polyfit(log_k, log_a, 1)
    gamma_fit = -coeffs[0]
    C_fit = np.exp(coeffs[1])

    predicted = coeffs[0] * log_k + coeffs[1]
    ss_res = np.sum((log_a - predicted) ** 2)
    ss_tot = np.sum((log_a - np.mean(log_a)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return gamma_fit, C_fit, r_squared, int(mask.sum())


def main():
    parser = argparse.ArgumentParser(
        description="Paper 3: Compute A(K) area decay and fit γ for each g-function.")
    parser.add_argument("--resolution", type=int, default=1000,
                        help="Grid resolution for area measurement (default 1000)")
    parser.add_argument("--iter", type=int, default=500,
                        help="Max iterations for convergence test")
    parser.add_argument("--escape", type=float, default=50.0)
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    area_csv = os.path.join(args.outdir, "area_decay_sweep.csv")
    gamma_csv = os.path.join(args.outdir, "gamma_fit_results.csv")

    print(f"=== Paper 3: A(K) Area Decay Sweep (Adaptive Viewport) ===")
    print(f"    Resolution: {args.resolution}×{args.resolution}")
    print(f"    K values: {len(K_SWEEP_SMALL)} from {K_SWEEP_SMALL[0]} to {K_SWEEP_SMALL[-1]}")
    print(f"    g-functions: {len(G_FUNCTIONS)}\n")

    all_results = []
    area_by_g = {g[0]: ([], []) for g in G_FUNCTIONS}

    t_start = time.time()
    total = len(G_FUNCTIONS) * len(K_SWEEP_SMALL)
    count = 0

    for g_name, g_func, g_label in G_FUNCTIONS:
        print(f"\n  g(y) = {g_label}")
        for K in K_SWEEP_SMALL:
            count += 1
            t0 = time.time()
            A, vp = measure_area_adaptive(K, g_func, g_name,
                                          args.resolution, args.iter, args.escape)
            dt = time.time() - t0
            all_results.append((g_name, g_label, K, A, vp))
            area_by_g[g_name][0].append(K)
            area_by_g[g_name][1].append(A)
            print(f"    [{count:>3d}/{total}] K={K:>8,d}  A(K)={A:.8e}  "
                  f"vp={vp:.4e}  ({dt:.1f}s)")

    # ── Save area CSV ────────────────────────────────────────────────────────
    with open(area_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["g_name", "g_label", "K", "A_K", "viewport_area"])
        for row in all_results:
            w.writerow(row)
    print(f"\n  Saved: {area_csv}")

    # ── Fit γ ────────────────────────────────────────────────────────────────
    print(f"\n=== γ Exponent Fitting ===\n")
    gamma_results = []

    for g_name, _, g_label in G_FUNCTIONS:
        k_vals, a_vals = area_by_g[g_name]
        gamma_fit, C_fit, r_sq, n_pts = fit_gamma(k_vals, a_vals)
        gamma_pred = GAMMA_PREDICTED.get(g_name, None)

        if gamma_fit is not None:
            error = abs(gamma_fit - gamma_pred) if gamma_pred else None
            pct_err = 100 * error / gamma_pred if error is not None and gamma_pred else None
            print(f"  {g_label:<20s}  γ_fit={gamma_fit:.4f}  γ_pred={gamma_pred:.4f}"
                  f"  err={pct_err:.1f}%  R²={r_sq:.6f}  n={n_pts}")
        else:
            pct_err = None
            error = None
            print(f"  {g_label:<20s}  INSUFFICIENT DATA (n={n_pts})")

        gamma_results.append({
            "g_name": g_name,
            "g_label": g_label,
            "gamma_fit": gamma_fit,
            "gamma_predicted": gamma_pred,
            "C_fit": C_fit,
            "r_squared": r_sq,
            "abs_error": error,
            "pct_error": pct_err,
            "n_points": n_pts,
        })

    with open(gamma_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gamma_results[0].keys())
        w.writeheader()
        w.writerows(gamma_results)
    print(f"\n  Saved: {gamma_csv}")

    elapsed = time.time() - t_start
    print(f"\n=== Done: {elapsed:.0f}s ({elapsed/60:.1f} min) ===")


if __name__ == "__main__":
    main()
