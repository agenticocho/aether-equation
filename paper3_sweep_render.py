#!/usr/bin/env python3
"""paper3_sweep_render.py — Render the Aether Set for 7 g-functions × 6 K values.

Extends aether_fractal_renderer.py for the Paper 3 generalized iteration:
    z_{n+1} = K · z_n · g(Im(z_n)) + c

Test functions:
    Class A (power-law):  g(y) = |y|^(1/2),  |y|^(3/4),  |y|^1,  |y|^2
    Class B (smooth):     g(y) = exp(-|y|),   tanh(|y|),  sigmoid(|y|)

K values: 2, 5, 10, 20, 50, 100

Total renders: 7 × 6 = 42 at 4K resolution (3840×2160).

Usage:
    python3 paper3_sweep_render.py [--width 3840] [--height 2160] [--iter 500]
                                    [--outdir renders_paper3]

Builds on aether_fractal_renderer.py — reuses colorize() and add_annotations().

License: MIT
"""

import argparse
import os
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Import existing colorization from the repo ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aether_fractal_renderer import colorize


# ── g-function definitions ───────────────────────────────────────────────────

def g_power(y, alpha):
    """g(y) = |y|^alpha.  For alpha < 1 this vanishes at y=0."""
    return np.power(np.abs(y) + 1e-30, alpha)  # epsilon avoids 0^0

def g_exp_neg(y):
    """g(y) = exp(-|y|).  Class B — smooth, g(0)=1, g'(0)=0 from right."""
    return np.exp(-np.abs(y))

def g_tanh(y):
    """g(y) = tanh(|y|).  Class B — vanishing order α=1."""
    return np.tanh(np.abs(y))

def g_sigmoid(y):
    """g(y) = sigmoid(|y|) = 1/(1+exp(-|y|)).  Class A — g(0)=0.5 > 0."""
    return 1.0 / (1.0 + np.exp(-np.abs(y)))


# Registry: (label, function, short_name)
G_FUNCTIONS = [
    ("power_alpha_0.5",  lambda y: g_power(y, 0.5),   "|y|^{1/2}"),
    ("power_alpha_0.75", lambda y: g_power(y, 0.75),  "|y|^{3/4}"),
    ("power_alpha_1.0",  lambda y: g_power(y, 1.0),   "|y|^1"),
    ("power_alpha_2.0",  lambda y: g_power(y, 2.0),   "|y|^2"),
    ("exp_neg_abs",      g_exp_neg,                    "exp(-|y|)"),
    ("tanh_abs",         g_tanh,                       "tanh(|y|)"),
    ("sigmoid_abs",      g_sigmoid,                    "sigmoid(|y|)"),
]

K_VALUES = [2, 5, 10, 20, 50, 100]


# ── Generalized renderer ────────────────────────────────────────────────────

def render_aether_generalized(
    K, g_func, width, height, max_iter,
    escape_radius=50.0,
    bss_range=(-1.5, 1.5),
    delta_range=(-1.5, 1.5),
):
    """Render the generalized Aether Set: z_{n+1} = K · z · g(Im(z)) + c.

    Extends render_aether_fractal() from the existing renderer, replacing
    exp(Im(z)) with an arbitrary g(Im(z)).
    """
    re = np.linspace(bss_range[0], bss_range[1], width)
    im = np.linspace(delta_range[1], delta_range[0], height)

    C = re[np.newaxis, :] + 1j * im[:, np.newaxis]
    Z = np.zeros_like(C, dtype=np.complex128)

    escape = np.full((height, width), max_iter, dtype=np.int32)
    smooth = np.zeros((height, width), dtype=np.float64)

    for i in range(max_iter):
        mask = escape == max_iter
        if not mask.any():
            break

        Z_im = Z[mask].imag
        g_vals = g_func(Z_im)
        Z[mask] = K * Z[mask] * g_vals + C[mask]

        blown = mask & (np.abs(Z) > escape_radius)
        escape[blown] = i
        smooth[blown] = i + 1 - np.log2(
            np.log2(np.abs(Z[blown]) + 1e-10)
        )

    smooth[escape == max_iter] = max_iter
    return escape, smooth


def add_annotations_generalized(img_pil, K, g_label, bss_range, delta_range, width, height):
    """Annotate fractal render with K, g-function label, and axes."""
    draw = ImageDraw.Draw(img_pil)
    font_size = max(16, width // 160)
    title_size = max(24, width // 100)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", title_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
        title_font = font

    pad = 10
    shadow = (0, 0, 0)
    text_color = (220, 220, 220)

    title = f"Aether Set  K={K}, g(y)={g_label}"
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.text((pad + dx, pad + dy), title, font=title_font, fill=shadow)
    draw.text((pad, pad), title, font=title_font, fill=(255, 215, 0))

    eq = f"z(n+1) = {K} * z * g(Im(z)) + c"
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.text((pad + dx, pad + title_size + 8 + dy), eq, font=font, fill=shadow)
    draw.text((pad, pad + title_size + 8), eq, font=font, fill=text_color)

    bss_label = f"BSS [{bss_range[0]}, {bss_range[1]}]"
    delta_label = f"delta [{delta_range[0]}, {delta_range[1]}]"
    draw.text((width // 2 - 100, height - font_size - pad),
              bss_label, font=font, fill=text_color)
    draw.text((pad, height // 2), delta_label, font=font, fill=text_color)

    legend_y = height - font_size * 3 - pad
    draw.rectangle(
        [width - 360, legend_y - 5, width - pad, legend_y + font_size * 2 + 10],
        fill=(8, 12, 28))
    draw.text((width - 350, legend_y),
              "Dark: stable (convergent)", font=font, fill=(100, 140, 200))
    draw.text((width - 350, legend_y + font_size + 4),
              "Color: escape velocity (divergent)", font=font, fill=(40, 180, 120))

    # BSS=0 axis
    zero_x = int((0 - bss_range[0]) / (bss_range[1] - bss_range[0]) * width)
    if 0 < zero_x < width:
        for y in range(0, height, 6):
            draw.line([(zero_x, y), (zero_x, min(y + 3, height))],
                      fill=(255, 255, 255), width=1)
        draw.text((zero_x + 4, pad + title_size + font_size + 16),
                  "BSS=0", font=font, fill=(255, 255, 255))

    return img_pil


def main():
    parser = argparse.ArgumentParser(
        description="Paper 3 sweep: render Aether Set for all g-functions × K values.")
    parser.add_argument("--width", type=int, default=3840, help="Image width (default 4K=3840)")
    parser.add_argument("--height", type=int, default=2160, help="Image height (default 4K=2160)")
    parser.add_argument("--iter", type=int, default=500, help="Max iterations")
    parser.add_argument("--escape", type=float, default=50.0, help="Escape radius")
    parser.add_argument("--outdir", type=str, default="renders_paper3", help="Output directory")
    parser.add_argument("--bss-min", type=float, default=-1.5)
    parser.add_argument("--bss-max", type=float, default=1.5)
    parser.add_argument("--delta-min", type=float, default=-1.5)
    parser.add_argument("--delta-max", type=float, default=1.5)
    parser.add_argument("--palette", type=str, default="default",
                        help="Color palette: default or sunset")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    bss_range = (args.bss_min, args.bss_max)
    delta_range = (args.delta_min, args.delta_max)

    total = len(G_FUNCTIONS) * len(K_VALUES)
    count = 0
    t_start = time.time()

    print(f"=== Paper 3 Sweep: {total} renders at {args.width}x{args.height} ===")
    print(f"    g-functions: {len(G_FUNCTIONS)},  K values: {K_VALUES}")
    print(f"    Output: {args.outdir}/\n")

    for g_name, g_func, g_label in G_FUNCTIONS:
        for K in K_VALUES:
            count += 1
            fname = f"aether_K{K}_{g_name}.png"
            outpath = os.path.join(args.outdir, fname)

            print(f"[{count}/{total}] K={K:>3d}  g={g_label:<16s} -> {fname}")
            t0 = time.time()

            escape, smooth = render_aether_generalized(
                K, g_func, args.width, args.height, args.iter,
                args.escape, bss_range, delta_range)

            pixels = colorize(escape, smooth, args.iter, palette=args.palette)
            img = Image.fromarray(pixels)
            img = add_annotations_generalized(
                img, K, g_label, bss_range, delta_range, args.width, args.height)
            img.save(outpath, "PNG", optimize=True)

            in_set = np.sum(escape == args.iter)
            total_px = args.width * args.height
            pct = 100 * in_set / total_px
            fsize = os.path.getsize(outpath) / (1024 * 1024)

            print(f"       {time.time() - t0:.1f}s  |  {fsize:.1f} MB  |  "
                  f"stable: {in_set:,}/{total_px:,} ({pct:.4f}%)")

    elapsed = time.time() - t_start
    print(f"\n=== Done: {total} renders in {elapsed:.0f}s ({elapsed/60:.1f} min) ===")


if __name__ == "__main__":
    main()
