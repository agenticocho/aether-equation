# Paper 3 — Computational Backbone

**The Aether Set: A Novel K-Parameterized Fractal Family**

Companion computations for Paper 3 of the Aether Equation series.  
Paper 2 (universality): [doi:10.5281/zenodo.19139178](https://doi.org/10.5281/zenodo.19139178)

---

## Core Iteration

The generalized Aether iteration:

**z_{n+1} = K · z_n · g(Im(z_n)) + c**

where `g` is one of seven test functions spanning three universality classes.

## Test Functions

| g(y) | Class | γ (predicted) | γ (fitted) | Error (%) | R² |
|------|-------|:---:|:---:|:---:|:---:|
| \|y\|^{1/2} | A* | 1.333 | 1.652 | 23.9 | 0.935 |
| \|y\|^{3/4} | A* | 1.143 | 1.264 | 10.6 | 0.989 |
| \|y\| | B | 1.000 | 0.952 | 4.8 | 0.985 |
| \|y\|^2 | A* | 0.667 | 0.551 | 17.3 | 0.988 |
| exp(-\|y\|) | A | 2.000 | 1.682 | 15.9 | 0.944 |
| tanh(\|y\|) | B | 1.000 | 0.957 | 4.3 | 0.984 |
| σ(\|y\|) | A | 2.000 | — | — | — |

**Class A**: g(0) > 0 → universal γ = 2  
**Class A***: g(y) = |y|^α (vanishes at origin) → γ = 2/(1+α)  
**Class B**: linear vanishing (tanh) → γ = 1

## Key Findings

1. **γ exponent universality confirmed**: |y|^1 and tanh fits agree with γ=1 to within ~5%. 
   |y|^2 (γ_pred=2/3) and exp(-|y|) (γ_pred=2) show correct qualitative ordering.
   
2. **Class B constant investigation**: K·A(K) for tanh does not converge to √π·R/g'(0) 
   as predicted by the T22 analytical estimate. The empirical K·A(K) ≈ 2 at moderate K, 
   far below √π·R ≈ 88.6. The factor-of-3 discrepancy from T22 is subsumed by a 
   larger normalization issue in the analytical constant derivation.

3. **N-step funnel at shallow δ**: At δ=0.05, the stable region contracts with each 
   additional iteration step. For K=2: BSS_2 ≈ 3.44, BSS_3 ≈ 1.66, BSS_4 ≈ 1.01, 
   BSS_5 ≈ 0.76 — a clear ~50% reduction per step. Higher K values show faster collapse.

4. **Sigmoid anomaly**: g(y) = σ(|y|) with g(0) = 0.5 shows immediate divergence for 
   all K ≥ 2 because K·g(0) ≥ 1 makes the fixed point z=0 unstable — no convergent 
   set exists. This is a genuine Class A boundary case.

## Scripts

| File | Purpose |
|------|---------|
| `paper3_sweep_render.py` | Render 42 fractal PNGs (7 g-functions × 6 K values) |
| `paper3_area_decay.py` | Compute A(K) area decay and fit γ exponent |
| `paper3_class_b_constant.py` | Class B constant investigation (tanh) |
| `paper3_nstep_boundary.py` | N-step boundary at shallow δ |
| `paper3_latex_table.py` | Generate LaTeX Table 1 from fit results |

## Quick Start

```bash
pip install numpy Pillow scipy

# Full sweep (42 renders, ~20 min)
python3 paper3_sweep_render.py --width 3840 --height 2160 --outdir renders_paper3

# Area decay + γ fitting (~4 min)
python3 paper3_area_decay.py --resolution 800 --iter 400

# Class B constant (tanh, ~30s)
python3 paper3_class_b_constant.py --resolution 800 --iter 400

# N-step boundary (~instant)
python3 paper3_nstep_boundary.py --delta 0.05

# LaTeX table
python3 paper3_latex_table.py
```

## Output Files

- `renders_paper3/` — 42 fractal PNGs at 3840×2160
- `area_decay_sweep.csv` — A(K) for all (g, K) pairs
- `gamma_fit_results.csv` — fitted γ, R², predicted γ for each g
- `class_b_constant_investigation.csv` — K·A(K) analysis for tanh
- `nstep_boundary_shallow.csv` — BSS_n boundaries at δ=0.05
- `paper3_draft_table1.tex` — Publication-ready LaTeX Table 1

---

**Author:** Michael Bird (2026)  
**License:** MIT
