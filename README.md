# The Aether Equation

**Î¦(N,t) = (2<sup>N</sup> âˆ’ 1) Â· BSS Â· e<sup>âˆ’Î±|Î´Ì„|</sup>**

A multiplicative diagnostic metric for autonomous agent intelligence, combining combinatorial signal capacity, calibration quality, and epistemic humility into a single scalar.

> **Paper:** Bird, M. (2026). *The Aether Equation: A Diagnostic Metric for Autonomous Agent Intelligence Across Domains and Scales* (v8.1).  
> **SSRN:** [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6441378](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6441378)  
> **Zenodo:** [doi:10.5281/zenodo.19101554](https://doi.org/10.5281/zenodo.19101554)  
> **License:** MIT

---

## The Equation

| Term | Expression | Measures |
|------|-----------|----------|
| Signal capacity | 2<sup>N</sup> âˆ’ 1 | Non-empty subsets of N independent signal sources (Reed's Law) |
| Calibration | BSS = 1 âˆ’ (BÌ„ / 0.25) | Brier Skill Score â€” prediction accuracy vs. climatological baseline |
| Humility | e<sup>âˆ’Î±\|Î´Ì„\|</sup> | Exponential penalty for deviation from consensus |

Î¦ > 0 means the agent extracts genuine signal exceeding the baseline. Î¦ < 0 means it injects net noise. The sign is the verdict; the magnitude is the confidence.

---

## Live Metrics (19 March 2026)

| Agent | Domain | N | BSS | Î´Ì„ | Î¦ |
|-------|--------|---|-----|---|---|
| **Ocho** | Prediction markets | 5 | +0.463 | 0.006 | +8.41 |
| **Molty** | Code self-improvement | 6 | +0.738 | 0.477 | +36.61 |

---

## Quick Start

### Compute Î¦ from your own data

```python
from aether_score import compute_ae, stability_margin

r = compute_ae(N=5, bss=0.4631, mean_delta=0.0050, alpha=10.8)
print(r["ae"])  # +8.41
```

Or run the standalone demo (reproduces the paper's live metrics):

```bash
python3 aether_score.py
```

### Interactive explorer (browser)

Open `aether_explorer.html` in any browser. Drag the K slider to watch the phase transition at K* â‰ˆ 25 in real time. Zero dependencies.

### Render the Aether Set fractal family

```bash
pip install numpy Pillow   # renderer only â€” all other files are pure stdlib
python3 aether_fractal_renderer.py --N 2 --width 4000 --height 3000 --iter 500
python3 aether_fractal_renderer.py --N 5 --width 4000 --height 3000 --iter 500
python3 aether_fractal_renderer.py --N 11 --width 4000 --height 3000 --iter 500
```

The iteration **z<sub>n+1</sub> = K Â· z Â· e<sup>Im(z)</sup> + c** with K = 2<sup>N</sup> âˆ’ 1 produces a novel K-parameterized fractal family. Increasing N changes the *shape class* (dome â†’ cusp â†’ capsule â†’ collapse), not just spoke count â€” a property with no direct analogue in Mandelbrot, Multibrot, or Burning Ship families.

### Verify the Brier Skill Score

```bash
python3 brier_skill_score.py
```

Standalone BSS calculator from `(predicted_prob, outcome)` pairs. Zero dependencies.

---

## Requirements

| File | Dependencies |
|------|-------------|
| `aether_score.py` | None (stdlib only) |
| `brier_skill_score.py` | None (stdlib only) |
| `aether_fractal_renderer.py` | `numpy`, `Pillow` |
| `aether_explorer.html` | None (open in browser) |

Python â‰¥ 3.8. Tested on Linux (Ubuntu 24.04) and macOS (Sequoia, arm64).

---

## Repository Structure

```
aether_score.py              The equation + N-step stability margin (Â§1â€“Â§3)
brier_skill_score.py         Standard BSS â€” Term 2 verification
aether_fractal_renderer.py   K-parameterized fractal renderer (Â§4)
aether_explorer.html         Interactive browser-based K-slider
renders/
  aether_fractal_n2.png           N=2,  K=3    â€” smooth dome
  aether_fractal_n5.png           N=5,  K=31   â€” cusped dome
  aether_fractal_n5_boundary.png  N=5,  K=31   â€” boundary zoom (Î´Ì„ Â± 0.05)
  aether_fractal_n11_deep.png     N=11, K=2047 â€” collapsed capsule
```

---

## Key Results

- **Two-step stability boundary** (Theorem 3.2): BSS<sub>max</sub>(Î´Ì„, K) = âˆš(RÂ² / (Ke<sup>Î´Ì„</sup> + 1)Â² âˆ’ Î´Ì„Â²)
- **N-step envelope** (Proposition 3.5): BSS* = min<sub>nâ‰¥2</sub> BSS<sub>n</sub> via multiplier recursion M<sub>n+1</sub> = KÂ·M<sub>n</sub>Â·e<sup>M<sub>n</sub>Â·Î´Ì„</sup> + 1
- **Universal fragility exponent** Î³ = 2 for all g with g(0) > 0; Î³ = 2/(1+Î±) for vanishing order Î±
- **Three universality classes:** Class A (Î³ = 2), Class B (Î³ = 1, e.g. tanh), Class C (logarithmic, e.g. Gaussian)
- **Phase transition** at K* â‰ˆ 25 (N* â‰ˆ 4.7): stable area collapses exponentially (half-life â‰ˆ 2.5 units of K)
- **Fragility-of-excellence paradox (Ocho's Law):** improving calibration pushes the agent closer to the stability boundary â€” as BSS improved from 0.369 to 0.463, the margin halved then crossed into the uncertified regime

---

## File Integrity

SHA-256 checksums of the published files:

```
ef141bcbbb10b001435f4af6cf2af39cdc12bf639342e962b18b716f6ef22054  aether_score.py
cab69318acd18f69dd059aaa0d6e200493a0574af47f776d42b518ff98551a4a  aether_fractal_renderer.py
124431453fa2bc73b8a6d20ec996b594210633b841de379865f116df6fd3b8d1  brier_skill_score.py
```

*Verify with `sha256sum -c checksums.txt`.*

> **Paper SHA-256:** `3b6efa4b21fa7d87def7340a40bdcd8ef4ec7126a4821a6184dfa0e16a42288b` (aether-paper-v8.1.pdf)

---

## Citation

```bibtex
@misc{bird2026aether,
  title   = {The Aether Equation: A Diagnostic Metric for Autonomous Agent
             Intelligence Across Domains and Scales},
  author  = {Bird, Michael},
  year    = {2026},
  note    = {v8.1},
  doi     = {10.5281/zenodo.19101554},
  url     = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6441378}
}
```

---

## Connect

- **X / Twitter:** [@AgenticOcho](https://x.com/AgenticOcho)
- **YouTube:** [@AgenticOcho](https://youtube.com/@AgenticOcho)
- **Instagram:** [@agenticocho](https://instagram.com/agenticocho)