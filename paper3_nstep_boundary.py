#!/usr/bin/env python3
"""paper3_nstep_boundary.py — Open Problem: N-step boundary at shallow δ.

Compute the true stable region at δ=0.05 for iteration steps n=2,3,4,5.
Compare to the two-step analytical wall from Theorem 3.2.

Method:
    For each K in a log-spaced sweep:
    1. Compute the analytical BSS_n boundary via the M_n recursion
       (reusing bss_nstep from aether_score.py).
    2. Numerically verify by iterating the map n times and checking |z_n| < R.
    3. Report BSS_n for each n, and the ratio BSS_n / BSS_2 (convergence envelope).

The shallow regime (δ ~ 0) is where higher-step boundaries diverge most from
the 2-step wall, revealing the "N-step funnel" structure.

Usage:
    python3 paper3_nstep_boundary.py [--delta 0.05] [--outdir .]

License: MIT
"""

import argparse
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aether_score import bss_nstep, R_DEFAULT


# ── K sweep ──────────────────────────────────────────────────────────────────
K_SWEEP = [2, 3, 5, 7, 10, 15, 20, 31, 50, 75, 100, 150, 200,
           300, 500, 750, 1000, 2000, 5000, 10000, 50000, 100000]


def numerical_bss_boundary(delta, K, n_steps, R=10.0, bss_resolution=10000):
    """Numerically find the max BSS for which the orbit stays bounded after n steps.

    Bisection search: for a given δ, find the largest BSS such that
    z_n stays within radius R after n iterations starting from z_0 = 0,
    c = BSS + i·δ.
    """
    lo, hi = 0.0, R  # BSS range

    for _ in range(60):  # bisection iterations
        mid = (lo + hi) / 2.0
        c = complex(mid, delta)
        z = complex(0, 0)
        escaped = False

        for step in range(n_steps):
            # Use exp(Im(z)) as the canonical g-function for comparison
            z = K * z * math.exp(z.imag) + c
            if abs(z) > R or math.isnan(z.real) or math.isnan(z.imag):
                escaped = True
                break

        if escaped:
            hi = mid
        else:
            lo = mid

    return (lo + hi) / 2.0


def main():
    parser = argparse.ArgumentParser(
        description="Paper 3 OP: N-step boundary at shallow δ.")
    parser.add_argument("--delta", type=float, default=0.05,
                        help="Shallow δ value (default 0.05)")
    parser.add_argument("--R", type=float, default=R_DEFAULT,
                        help="Escape radius")
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "nstep_boundary_shallow.csv")

    delta = args.delta
    R = args.R
    n_steps = [2, 3, 4, 5]

    print(f"=== Paper 3 OP: N-step Boundary at Shallow δ ===")
    print(f"    δ = {delta}")
    print(f"    R = {R}")
    print(f"    Steps: {n_steps}")
    print(f"    K range: {K_SWEEP[0]} .. {K_SWEEP[-1]:,}\n")

    # Use reflected delta for analytical (negative convention from §3a)
    delta_reflected = -abs(delta)

    results = []
    t_start = time.time()

    print(f"  {'K':>8s}", end="")
    for n in n_steps:
        print(f"  {'BSS_'+str(n)+'(anal)':>14s}  {'BSS_'+str(n)+'(num)':>14s}", end="")
    print(f"  {'ratio_3/2':>10s}  {'ratio_4/2':>10s}  {'ratio_5/2':>10s}")
    print("  " + "-" * 120)

    for K in K_SWEEP:
        row = {"K": K, "delta": delta}

        # Analytical boundaries
        bss_vals_anal = {}
        for n in n_steps:
            b = bss_nstep(delta_reflected, K, n, R)
            bss_vals_anal[n] = b
            row[f"bss_{n}_analytical"] = b

        # Numerical boundaries (using exp(Im(z)) canonical)
        bss_vals_num = {}
        for n in n_steps:
            b = numerical_bss_boundary(delta, K, n, R)
            bss_vals_num[n] = b
            row[f"bss_{n}_numerical"] = b

        # Ratios relative to 2-step
        for n in [3, 4, 5]:
            if bss_vals_anal[2] > 0:
                row[f"ratio_{n}_over_2"] = bss_vals_anal[n] / bss_vals_anal[2]
            else:
                row[f"ratio_{n}_over_2"] = None

        results.append(row)

        # Print row
        print(f"  {K:>8d}", end="")
        for n in n_steps:
            print(f"  {bss_vals_anal[n]:>14.6f}  {bss_vals_num[n]:>14.6f}", end="")

        ratios = []
        for n in [3, 4, 5]:
            r = row.get(f"ratio_{n}_over_2")
            ratios.append(f"{r:.6f}" if r is not None else "N/A")
        print(f"  {'  '.join(f'{r:>10s}' for r in ratios)}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n  ── Summary ──")
    valid = [r for r in results if r.get("bss_2_analytical", 0) > 0]
    if valid:
        for n in [3, 4, 5]:
            ratios = [r[f"ratio_{n}_over_2"] for r in valid
                      if r.get(f"ratio_{n}_over_2") is not None]
            if ratios:
                print(f"  BSS_{n}/BSS_2 range: [{min(ratios):.4f}, {max(ratios):.4f}]"
                      f"  mean: {sum(ratios)/len(ratios):.4f}")

    print(f"\n  At shallow δ={delta}, higher n-step boundaries should tighten.")
    print(f"  The ratio BSS_n/BSS_2 < 1 shows the 'N-step funnel' narrowing.")

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
