#!/usr/bin/env python3
"""aether_score.py — The Aether Equation: Diagnostic Metric for Agent Intelligence.

Computes Φ(N,t) = (2^N - 1) · BSS · e^{-α|δ̄|}

Where:
    N   = number of independent signal sources (observation dimensionality)
    BSS = Brier Skill Score, calibration quality (Term 2)
    δ̄   = mean signed deviation from consensus (contrarian exposure)
    α   = decay constant governing humility penalty (Term 3)

Also provides:
    bss_nstep()       — exact n-step inner boundary via M_n recursion (T27)
    stability_margin() — distance to 2-step and 3-step analytical walls (T27b)
    brier_skill_score() — standard BSS from (prediction, outcome) pairs

Reference: "The Aether Equation" (2026), SSRN [TBD]
License: MIT

Changelog:
    T27b (17 Mar 2026):
        - stability_margin() now returns a dict (was a 3-tuple).
        - BOUNDARY3WARNING requires m_3step > 0 (T27b fix).
        - rho_op=1.278 removed (T22 proved it was a viewport artifact).
    T27 (17 Mar 2026):
        - Replaced single-step boundary formula with exact N-step M_n recursion.
        - Added bss_nstep(), delta_cross, regime to public API.
        - 4-tier status: SAFE / CAUTION / BOUNDARY3WARNING / OUTSIDE.
"""

import math
from typing import Dict, List, Optional, Tuple

# ── Paper constants ──────────────────────────────────────────────────────────
ALPHA_DEFAULT = 10.8   # Empirical decay constant (§2, Table 1)
R_DEFAULT     = 10.0   # Iteration escape radius (§3a)
BSS_BASELINE  = 0.25   # Climatological Brier score (coin-flip reference)

# ── Term 1: Combinatorial signal space ───────────────────────────────────────
def signal_capacity(N: int) -> int:
    """2^N - 1: number of non-empty subsets of N signal sources."""
    return 2**N - 1


# ── Term 2: Calibration quality ──────────────────────────────────────────────
def brier_skill_score(
    predictions: List[Tuple[float, float]],
    baseline: float = BSS_BASELINE,
) -> Tuple[float, float, int]:
    """Compute BSS from a list of (predicted_prob, binary_outcome) pairs.

    Returns (bss, mean_brier, n).
    """
    if not predictions:
        return 0.0, 0.0, 0
    n = len(predictions)
    total = sum((p - o) ** 2 for p, o in predictions)
    mean_brier = total / n
    bss = 1.0 - (mean_brier / baseline)
    return bss, mean_brier, n


# ── Term 3: Contrarian decay ─────────────────────────────────────────────────
def contrarian_decay(mean_delta: float, alpha: float = ALPHA_DEFAULT) -> float:
    """e^{-α|δ̄|}: exponential penalty for deviation from consensus."""
    return math.exp(-alpha * abs(mean_delta))


# ── The Equation ─────────────────────────────────────────────────────────────
def compute_ae(
    N: int,
    bss: float,
    mean_delta: float,
    alpha: float = ALPHA_DEFAULT,
) -> dict:
    """Compute Φ(N,t) = (2^N - 1) · BSS · e^{-α|δ̄|}.

    Args:
        N:          Number of independent signal sources.
        bss:        Brier Skill Score (Term 2). Range (-∞, 1].
        mean_delta: Mean absolute deviation |δ̄| from consensus.
        alpha:      Decay constant (default 10.8).

    Returns:
        dict with keys: ae, K, bss, decay, mean_delta, alpha, N.
    """
    K = signal_capacity(N)
    decay = contrarian_decay(mean_delta, alpha)
    ae = K * bss * decay
    return {
        "ae":         round(ae, 4),
        "K":          K,
        "bss":        round(bss, 4),
        "decay":      round(decay, 4),
        "mean_delta": round(mean_delta, 4),
        "alpha":      alpha,
        "N":          N,
    }


# ── N-step boundary (T27) ─────────────────────────────────────────────────────
def bss_nstep(delta_bar: float, K: int, n: int, R: float = R_DEFAULT) -> float:
    """Exact n-step inner boundary via M_n recursion (T27 Theorem).

    Recursion:
        M_1     = 1
        M_{k+1} = K * M_k * exp(M_k * delta_bar) + 1

    BSS_n = sqrt(R^2 / M_n^2 - delta_bar^2)  if > 0, else 0.

    Theorem 3.2 (2-step) is the special case n=2.

    Args:
        delta_bar: Reflected bias δ̄ (negative convention, §3a).
        K:         Combinatorial capacity 2^N - 1.
        n:         Number of iteration steps (2 = Thm 3.2, 3 = T27).
        R:         Escape radius (default 10.0).

    Returns:
        Inner boundary BSS value; 0.0 if wall collapses.
    """
    M = 1.0
    for _ in range(n - 1):
        M = K * M * math.exp(M * delta_bar) + 1.0
        if M > 1e15 or math.isnan(M) or math.isinf(M):
            return 0.0
    inner = R**2 / M**2 - delta_bar**2
    return math.sqrt(inner) if inner > 0 else 0.0


# ── Stability margin (T27b) ───────────────────────────────────────────────────
def stability_margin(
    bss: float,
    delta_bar: float,
    K: int,
    R: float = R_DEFAULT,
) -> Dict:
    """Distance to the 2-step and 3-step analytical walls (T27b).

    Computes exact boundaries via bss_nstep() M_n recursion and classifies
    the agent into one of four status tiers.

    Status tiers (T27b — corrected logic):
        SAFE              m_inner > 0.15
        CAUTION           0 < m_inner ≤ 0.15
        BOUNDARY3WARNING  m_inner ≤ 0  AND  m_3step > 0   (between walls)
        OUTSIDE           m_inner ≤ 0  AND  (m_3step ≤ 0 OR bss_3step == 0)

    Args:
        bss:       Agent's current Brier Skill Score.
        delta_bar: Reflected bias δ̄ (negative convention, §3a).
        K:         Combinatorial capacity 2^N - 1.
        R:         Escape radius (default 10.0).

    Returns:
        dict with keys:
            bss_2step   — Theorem 3.2 inner bound
            bss_3step   — T27 3-step wall (may be 0.0 at shallow δ̄)
            m_inner     — margin vs 2-step wall (>0: inside, <0: outside)
            m_3step     — margin vs 3-step wall (None if bss_3step == 0)
            status      — 'SAFE' / 'CAUTION' / 'BOUNDARY3WARNING' / 'OUTSIDE'
            regime      — 'deep_tower' or 'shallow'
            delta_cross — leading-order crossover depth ≈ -ln(K)/K  (T27)
    """
    bss_2 = bss_nstep(delta_bar, K, 2, R)
    bss_3 = bss_nstep(delta_bar, K, 3, R)

    m_inner = (1.0 - bss / bss_2) if bss_2 > 0 else None
    m_3step = (1.0 - bss / bss_3) if bss_3 > 0 else None

    delta_cross = -math.log(K) / K        # leading-order asymptotics (T27)
    regime = "deep_tower" if delta_bar < delta_cross else "shallow"

    # T27b fix: BOUNDARY3WARNING requires m_3step > 0 (inside 3-step wall).
    if m_inner is not None and m_inner > 0.15:
        status = "SAFE"
    elif m_inner is not None and m_inner > 0.0:
        status = "CAUTION"
    elif (m_inner is not None and m_inner <= 0.0
          and m_3step is not None and m_3step > 0.0):
        status = "BOUNDARY3WARNING"
    else:
        status = "OUTSIDE"

    return {
        "bss_2step":   bss_2,
        "bss_3step":   bss_3,
        "m_inner":     m_inner,
        "m_3step":     m_3step,
        "status":      status,
        "regime":      regime,
        "delta_cross": delta_cross,
    }


# ── Example: reproduce Table 2 from the paper ────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Φ(N,t) = (2^N - 1) · BSS · e^{-α|δ̄|}")
    print("  The Aether Equation — reference implementation")
    print("=" * 60)

    # Published summary statistics (aether-paper-v6, 17 Mar 2026, n=94)
    examples = [
        ("Post-O48 Combined",     5,  0.4631,  0.0050, 10.8),
        ("Kalshi (isolated)",     5,  0.2062,  0.0500, 10.8),
        ("Polymarket (isolated)", 5,  0.5698,  0.0493, 10.8),
        ("Cross-venue (N=2)",     2,  0.4631,  0.0050, 10.8),
    ]

    for label, N, bss, delta, alpha in examples:
        r = compute_ae(N, bss, delta, alpha)
        sign = "POSITIVE" if r["ae"] > 0 else "NEGATIVE"
        print(f"\n  {label}  (n=94 / v6):")
        print(f"    K     = {r['K']},  BSS = {r['bss']:+.4f}")
        print(f"    decay = {r['decay']:.4f},  Φ  = {r['ae']:+.4f}  [{sign}]")

    # Stability margin for headline cohort (T27b)
    print("\n" + "-" * 60)
    print("  Stability Margin  (T27b — N-step boundary)")
    print("-" * 60)
    K_demo  = signal_capacity(5)            # K = 31 for N=5
    dbar    = -abs(0.0050)                  # reflected convention (§3a)
    bss_demo = 0.4631

    sm = stability_margin(bss_demo, dbar, K_demo)

    icons = {
        "SAFE":             "🟢",
        "CAUTION":          "🟡",
        "BOUNDARY3WARNING": "🟠",
        "OUTSIDE":          "🔴",
    }

    print(f"  δ̄  (reflected) = {dbar:+.4f}  [{sm['regime']}]")
    print(f"  δ_cross        = {sm['delta_cross']:+.4f}  (T27 crossover depth)")
    print(f"  BSS            = {bss_demo:.4f}")
    print()
    print(f"  BSS_2step = {sm['bss_2step']:.4f}  (Thm 3.2 inner)")

    if sm["m_inner"] is not None:
        print(f"  m_inner   = {sm['m_inner']:.4f}  ({sm['m_inner']*100:.1f}% from 2-step wall)")
    else:
        print("  m_inner   = outside analytical dome")

    print()
    b3 = sm["bss_3step"]
    m3 = sm["m_3step"]
    if b3 > 0 and m3 is not None and m3 > 0:
        print(f"  BSS_3step = {b3:.4f}  (T27 exact 3-step wall)")
        print(f"  m_3step   = {m3:.4f}  ({m3*100:.1f}% from 3-step wall)")
    elif b3 > 0 and m3 is not None and m3 <= 0:
        print(f"  BSS_3step = {b3:.4f}  (T27 exact 3-step wall)")
        print(f"  m_3step   = beyond 3-step wall  "
              f"(BSS is {bss_demo/b3:.1f}x the wall)")
    else:
        print(f"  BSS_3step = 0.0000  (3-step wall collapses at this δ̄, T27)")
        print(f"  m_3step   = n/a — beyond 3-step boundary")

    print()
    stat = sm["status"]
    print(f"  {icons.get(stat, '❓')} Status: {stat}")
    if stat == "OUTSIDE":
        print("  ⚠️  Agent is beyond both the 2-step AND 3-step analytical walls.")
        print("      Empirically stable but not orbit-certified.")
    elif stat == "BOUNDARY3WARNING":
        print("  ⚠️  Beyond 2-step wall but inside 3-step wall.")
    elif stat == "CAUTION":
        print("  ⚠️  Inside 2-step wall but margin < 15%.")
    else:
        print("  ✅  Comfortably inside 2-step analytical wall.")

    print()
    print("=" * 60)
