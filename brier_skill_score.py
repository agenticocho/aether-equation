#!/usr/bin/env python3
# Reference: "The Aether Equation" (2026), SSRN [link TBD]
# License: MIT
"""
brier_skill_score.py — Standard Brier Skill Score (Term 2 of the Aether Equation).

BSS = 1 - (mean_brier / baseline)

Where:
    mean_brier = (1/n) Σ (predicted_prob - outcome)²
    baseline   = 0.25 (climatological coin-flip reference)

BSS > 0  → better than guessing
BSS = 0  → equivalent to guessing
BSS < 0  → worse than guessing

Deps: none (stdlib only)

Usage:
    python3 brier_skill_score.py
"""

from typing import List, Tuple

BASELINE = 0.25


def brier_score(predicted: float, outcome: float) -> float:
    """Single-trade Brier score: (predicted_prob - binary_outcome)²."""
    return (predicted - outcome) ** 2


def brier_skill_score(
    predictions: List[Tuple[float, float]],
    baseline: float = BASELINE,
) -> Tuple[float, float, int]:
    """Compute BSS from a list of (predicted_prob, binary_outcome) pairs.

    Args:
        predictions: List of (p, o) where p ∈ [0,1] and o ∈ {0, 1}.
        baseline:    Reference Brier score (default 0.25, coin-flip).

    Returns:
        (bss, mean_brier, n)
    """
    if not predictions:
        return 0.0, 0.0, 0
    n = len(predictions)
    total = sum(brier_score(p, o) for p, o in predictions)
    mean_brier = total / n
    bss = 1.0 - (mean_brier / baseline)
    return bss, mean_brier, n


if __name__ == "__main__":
    print("=" * 50)
    print("  Brier Skill Score — Term 2 of the Aether Equation")
    print("  BSS = 1 - (mean_brier / 0.25)")
    print("=" * 50)

    examples = [
        ("Perfect forecaster",   [(0.9, 1), (0.1, 0), (0.8, 1), (0.2, 0)]),
        ("Coin-flip guessing",   [(0.5, 1), (0.5, 0), (0.5, 1), (0.5, 0)]),
        ("Confident and wrong",  [(0.9, 0), (0.1, 1), (0.8, 0), (0.2, 1)]),
        ("Slight edge (typical)", [(0.6, 1), (0.4, 0), (0.55, 1), (0.7, 1)]),
    ]

    for label, trades in examples:
        bss, mean_b, n = brier_skill_score(trades)
        sign = "+" if bss > 0 else ""
        status = "SKILL" if bss > 0 else ("BASELINE" if bss == 0 else "NO SKILL")
        print(f"\n  {label} (n={n}):")
        print(f"    mean Brier = {mean_b:.4f}")
        print(f"    BSS = {sign}{bss:.4f}  [{status}]")
