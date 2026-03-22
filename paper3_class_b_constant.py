#!/usr/bin/env python3
"""paper3_class_b_constant.py — Open Problem: Class B constant investigation.

For g(y) = tanh(|y|) at K = 10^1 .. 10^5:
    1. Measure A(K) with adaptive viewport
    2. Compute K · A(K) — the candidate Class B constant
    3. Compare to analytical prediction: 3√π · R / (K · g'(0))
    4. Diagnose the factor-of-3 discrepancy from T22

g'(0) for tanh: d/dy tanh(|y|) at y → 0+ = sech²(0) = 1.0

tanh is Class B (vanishing at origin), so A(K) ~ C / K, meaning K·A(K) → const.
The question: does the constant match √π·R/g'(0) or 3√π·R/g'(0)?

Usage:
    python3 paper3_class_b_constant.py [--resolution 1000] [--iter 500]

License: MIT
"""

import argparse
import csv
import math
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper3_sweep_render import g_tanh, render_aether_generalized


# ── K sweep for Class B investigation ────────────────────────────────────────
# Focus on moderate K where the set is measurable with pixel-counting
K_CLASS_B = [5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500,
             700, 1000, 1500, 2000, 3000, 5000, 7000, 10000]


def measure_area_adaptive(K, g_func, resolution, max_iter, escape_radius):
    """Compute A(K) with adaptive viewport for tanh (vanishing at 0).

    For tanh, the set scale is ~ R/K near the origin.
    """
    R = escape_radius
    scale = min(R / max(K * 0.3, 1), 2.0)
    scale = max(scale, 0.005)
    bss_range = (-scale, scale)
    delta_range = (-scale, scale)
    viewport_area = (2 * scale) ** 2

    escape, _ = render_aether_generalized(
        K, g_func, resolution, resolution, max_iter,
        escape_radius, bss_range, delta_range)

    convergent = np.sum(escape == max_iter)
    total = resolution * resolution
    return (convergent / total) * viewport_area, viewport_area


def main():
    parser = argparse.ArgumentParser(
        description="Paper 3 OP: Class B constant investigation (tanh).")
    parser.add_argument("--resolution", type=int, default=1000,
                        help="Grid resolution (default 1000)")
    parser.add_argument("--iter", type=int, default=500,
                        help="Max iterations")
    parser.add_argument("--escape", type=float, default=50.0)
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "class_b_constant_investigation.csv")

    R = args.escape
    g_prime_0 = 1.0  # sech²(0) = 1 for tanh

    print(f"=== Paper 3 OP: Class B Constant Investigation ===")
    print(f"    g(y) = tanh(|y|),  g'(0) = {g_prime_0}")
    print(f"    Resolution: {args.resolution}×{args.resolution}")
    print(f"    K range: {K_CLASS_B[0]} .. {K_CLASS_B[-1]:,}\n")

    # Analytical predictions (both hypotheses)
    # H1: K·A(K) → √π · R / g'(0)        (without factor of 3)
    # H2: K·A(K) → 3√π · R / g'(0)       (with factor of 3, from T22)
    sqrt_pi_R = math.sqrt(math.pi) * R / g_prime_0
    three_sqrt_pi_R = 3.0 * sqrt_pi_R

    print(f"    H1 (no factor 3):  K·A(K) → √π·R/g'(0)   = {sqrt_pi_R:.4f}")
    print(f"    H2 (factor 3):     K·A(K) → 3√π·R/g'(0)  = {three_sqrt_pi_R:.4f}\n")

    results = []
    t_start = time.time()

    for i, K in enumerate(K_CLASS_B):
        t0 = time.time()
        A, vp = measure_area_adaptive(K, g_tanh, args.resolution, args.iter, args.escape)
        KA = K * A
        dt = time.time() - t0

        ratio_h1 = KA / sqrt_pi_R if sqrt_pi_R > 0 else 0.0
        ratio_h2 = KA / three_sqrt_pi_R if three_sqrt_pi_R > 0 else 0.0

        results.append({
            "K": K,
            "A_K": A,
            "K_times_A_K": KA,
            "pred_sqrt_pi_R": sqrt_pi_R,
            "pred_3_sqrt_pi_R": three_sqrt_pi_R,
            "ratio_H1_no_3": ratio_h1,
            "ratio_H2_with_3": ratio_h2,
            "viewport_area": vp,
            "resolution": args.resolution,
            "max_iter": args.iter,
        })

        status = "✓" if A > 0 else "×"
        print(f"  [{i+1:>2d}/{len(K_CLASS_B)}] K={K:>8,d}  "
              f"A(K)={A:.6e}  K·A(K)={KA:.4f}  "
              f"H1={ratio_h1:.4f}  H2={ratio_h2:.4f}  "
              f"vp={vp:.4e}  {status} ({dt:.1f}s)")

    # ── Summary diagnostics ──────────────────────────────────────────────────
    valid = [r for r in results if r["A_K"] > 0]
    if len(valid) >= 3:
        # Focus on larger K values where asymptotics should hold
        high_k = [r for r in valid if r["K"] >= 50]
        if len(high_k) >= 3:
            analysis = high_k
            label = f"K≥50 ({len(high_k)} points)"
        else:
            analysis = valid
            label = f"all valid ({len(valid)} points)"

        ratios_h1 = [r["ratio_H1_no_3"] for r in analysis]
        ratios_h2 = [r["ratio_H2_with_3"] for r in analysis]

        mean_h1 = np.mean(ratios_h1)
        std_h1 = np.std(ratios_h1)
        mean_h2 = np.mean(ratios_h2)
        std_h2 = np.std(ratios_h2)

        print(f"\n  ── Diagnostic Summary ({label}) ──")
        print(f"  H1 (√π·R):   K·A(K)/pred = {mean_h1:.4f} ± {std_h1:.4f}")
        print(f"  H2 (3√π·R):  K·A(K)/pred = {mean_h2:.4f} ± {std_h2:.4f}")

        # KA values
        ka_vals = [r["K_times_A_K"] for r in analysis]
        mean_ka = np.mean(ka_vals)
        std_ka = np.std(ka_vals)
        print(f"  Mean K·A(K) = {mean_ka:.4f} ± {std_ka:.4f}")

        dev_h1 = abs(mean_h1 - 1.0)
        dev_h2 = abs(mean_h2 - 1.0)
        if dev_h1 < dev_h2:
            print(f"\n  FINDING: H1 (√π·R/g'(0)) fits better — factor of 3 is "
                  f"likely a T22 artifact.")
            print(f"  Empirical constant ≈ {mean_ka:.4f}, predicted √π·R = {sqrt_pi_R:.4f}")
        else:
            print(f"\n  FINDING: H2 (3√π·R/g'(0)) fits better — factor of 3 "
                  f"appears genuine.")
            print(f"  Empirical constant ≈ {mean_ka:.4f}, predicted 3√π·R = {three_sqrt_pi_R:.4f}")
    else:
        print(f"\n  WARNING: Only {len(valid)} valid data points — insufficient for diagnosis.")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\n  Saved: {csv_path}")

    elapsed = time.time() - t_start
    print(f"\n=== Done: {elapsed:.0f}s ({elapsed/60:.1f} min) ===")


if __name__ == "__main__":
    main()
