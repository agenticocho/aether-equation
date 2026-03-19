#!/usr/bin/env python3
# Reference: "The Aether Equation" (2026), SSRN [link TBD]
# License: MIT
"""
aether_fractal_renderer.py - Publication-quality Aether Set renders.
Iteration: z_{n+1} = K * z * exp(Im(z)) + c
where K = 2^N - 1 by default, c = BSS + i*delta (complex plane).

Deps: numpy, Pillow (pip install numpy Pillow)

Usage (canonical K = 2^N - 1):
  python3 aether_fractal_renderer.py --N 5 --width 4000 --height 3000 --iter 500
  python3 aether_fractal_renderer.py --N 2 --width 4000 --height 3000 --iter 500
  python3 aether_fractal_renderer.py --N 11 --width 4000 --height 3000 --iter 500

Override K directly (transition study):
  python3 aether_fractal_renderer.py --N 5 --K 24 --width 3000 --height 2250 --iter 1000 \
      --bss-min -0.25 --bss-max 0.25 --delta-min -0.6 --delta-max -0.15 \
      --output aether_K24_transition.png
"""

import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, sys, time


def render_aether_fractal(
    N,
    width,
    height,
    max_iter,
    escape_radius=50.0,
    bss_range=(-1.5, 1.5),
    delta_range=(-1.5, 1.5),
    override_K=None,
):
    # Allow explicit K override for transition experiments
    K = override_K if override_K is not None else (2**N - 1)

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

        # Aether iteration: K * z * exp(Im(z)) + c
        Z_im = Z.imag
        Z[mask] = K * Z[mask] * np.exp(Z_im[mask]) + C[mask]

        blown = mask & (np.abs(Z) > escape_radius)
        escape[blown] = i
        # Smooth coloring for escape velocity
        smooth[blown] = i + 1 - np.log2(
            np.log2(np.abs(Z[blown]) + 1e-10)
        )

    smooth[escape == max_iter] = max_iter
    return escape, smooth, K


def colorize(escape, smooth, max_iter, palette="default"):
    h, w = escape.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)

    in_set = escape == max_iter
    escaped = ~in_set

    # Stable points: dark background
    img[in_set] = [10, 18, 22]

    if escaped.any():
        vals = smooth[escaped]
        normed = np.sqrt(
            (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)
        )

        if palette == "sunset":
            r = np.clip(
                np.where(normed < 0.4, 30 + normed / 0.4 * 225, 255),
                0,
                255,
            )
            g = np.clip(
                np.where(
                    normed < 0.3,
                    normed / 0.3 * 140,
                    np.where(
                        normed < 0.6,
                        140 * (1 - (normed - 0.3) / 0.3),
                        0,
                    ),
                ),
                0,
                255,
            )
            b = np.clip(
                np.where(
                    normed < 0.5,
                    0,
                    np.where(
                        normed < 0.8,
                            (normed - 0.5) / 0.3 * 180,
                            180 + (normed - 0.8) / 0.2 * 75,
                    ),
                ),
                0,
                255,
            )
        else:
            r = np.clip(
                np.where(
                    normed < 0.5,
                    normed * 2 * 40,
                    40 + (normed - 0.5) * 2 * 215,
                ),
                0,
                255,
            )
            g = np.clip(
                np.where(
                    normed < 0.3,
                    normed / 0.3 * 180,
                    np.where(
                        normed < 0.7,
                        180 + (normed - 0.3) / 0.4 * 75,
                        220 + (normed - 0.7) / 0.3 * 35,
                    ),
                ),
                0,
                255,
            )
            b = np.clip(
                np.where(
                    normed < 0.4,
                    80 + normed / 0.4 * 120,
                    np.where(
                        normed < 0.8,
                        200 - (normed - 0.4) / 0.4 * 140,
                        60 + (normed - 0.8) / 0.2 * 195,
                    ),
                ),
                0,
                255,
            )

        img[escaped, 0] = r.astype(np.uint8)
        img[escaped, 1] = g.astype(np.uint8)
        img[escaped, 2] = b.astype(np.uint8)

    return img


def add_annotations(img_pil, N, K, bss_range, delta_range, width, height):
    draw = ImageDraw.Draw(img_pil)

    font_size = max(16, width // 160)
    title_size = max(24, width // 100)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            font_size,
        )
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            title_size,
        )
    except (OSError, IOError):
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/TTF/DejaVuSansMono.ttf", font_size
            )
            title_font = ImageFont.truetype(
                "/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf", title_size
            )
        except (OSError, IOError):
            font = ImageFont.load_default()
            title_font = font

    pad = 10
    shadow = (0, 0, 0)
    text_color = (220, 220, 220)

    title = f"The Aether Set  N={N}, K={K}"
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.text((pad + dx, pad + dy), title, font=title_font, fill=shadow)
    draw.text((pad, pad), title, font=title_font, fill=(255, 215, 0))

    eq = f"z(n+1) = {K} * z * exp(Im(z)) + c"
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.text(
            (pad + dx, pad + title_size + 8 + dy),
            eq,
            font=font,
            fill=shadow,
        )
    draw.text(
        (pad, pad + title_size + 8), eq, font=font, fill=text_color
    )

    bss_label = f"BSS [{bss_range[0]}, {bss_range[1]}]"
    delta_label = f"delta [{delta_range[0]}, {delta_range[1]}]"

    draw.text(
        (width // 2 - 100, height - font_size - pad),
        bss_label,
        font=font,
        fill=text_color,
    )
    draw.text((pad, height // 2), delta_label, font=font, fill=text_color)

    legend_y = height - font_size * 3 - pad
    draw.rectangle(
        [width - 360, legend_y - 5, width - pad, legend_y + font_size * 2 + 10],
        fill=(8, 12, 28),
    )
    draw.text(
        (width - 350, legend_y),
        "Dark: stable intelligence (convergent)",
        font=font,
        fill=(100, 140, 200),
    )
    draw.text(
        (width - 350, legend_y + font_size + 4),
        "Color: escape velocity (divergent)",
        font=font,
        fill=(40, 180, 120),
    )

    # BSS=0 axis
    zero_x = int(
        (0 - bss_range[0]) / (bss_range[1] - bss_range[0]) * width
    )
    if 0 < zero_x < width:
        for y in range(0, height, 6):
            draw.line(
                [(zero_x, y), (zero_x, min(y + 3, height))],
                fill=(255, 255, 255),
                width=1,
            )
        draw.text(
            (zero_x + 4, pad + title_size + font_size + 16),
            "BSS=0",
            font=font,
            fill=(255, 255, 255),
        )

    return img_pil


def main():
    parser = argparse.ArgumentParser(
        description="Render the Aether Set fractal."
    )
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--K", type=int, default=None,
                        help="Override K directly (default 2^N-1)")
    parser.add_argument("--width", type=int, default=4000)
    parser.add_argument("--height", type=int, default=3000)
    parser.add_argument("--iter", type=int, default=500)
    parser.add_argument("--escape", type=float, default=50.0)
    parser.add_argument("--bss-min", type=float, default=-1.5)
    parser.add_argument("--bss-max", type=float, default=1.5)
    parser.add_argument("--delta-min", type=float, default=-1.5)
    parser.add_argument("--delta-max", type=float, default=1.5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--zoom", action="store_true",
                        help="Boundary zoom view")
    parser.add_argument(
        "--palette",
        type=str,
        default="default",
        help="Color palette: default or sunset",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Skip all text annotations",
    )
    args = parser.parse_args()

    if args.zoom:
        args.bss_min, args.bss_max = -0.2, 0.3
        args.delta_min, args.delta_max = -0.3, 0.3

    # Compute K (allow override)
    K = 2**args.N - 1
    if args.K is not None:
        K = args.K

    outfile = (
        args.output
        or f"aether_fractal_n{args.N}{'_zoom' if args.zoom else ''}.png"
    )
    bss_range = (args.bss_min, args.bss_max)
    delta_range = (args.delta_min, args.delta_max)

    print("=== Aether Fractal Renderer ===")
    print(f" N={args.N} K={K} {args.width}x{args.height} iter={args.iter}")
    print(f" BSS range: {bss_range}  delta range: {delta_range}")
    print(f" Output: {outfile}")
    print(" Computing...", end=" ", flush=True)

    t0 = time.time()
    escape, smooth, K_used = render_aether_fractal(
        args.N,
        args.width,
        args.height,
        args.iter,
        args.escape,
        bss_range,
        delta_range,
        override_K=K,
    )
    t1 = time.time()
    print(f"done ({t1 - t0:.1f}s)")
    print(" Colorizing...", end=" ", flush=True)

    pixels = colorize(escape, smooth, args.iter, palette=args.palette)
    img = Image.fromarray(pixels)
    if not args.no_annotations:
        img = add_annotations(
            img,
            args.N,
            K_used,
            bss_range,
            delta_range,
            args.width,
            args.height,
        )
    t2 = time.time()
    print(f"done ({t2 - t1:.1f}s)")

    img.save(outfile, "PNG", optimize=True)
    fsize = os.path.getsize(outfile) / (1024 * 1024)
    print(f" Saved: {outfile} ({fsize:.1f} MB)")
    print(f" Total time: {t2 - t0:.1f}s")

    in_set = np.sum(escape == args.iter)
    total = args.width * args.height
    print(
        f" Convergent pixels: {in_set:,} / {total:,} "
        f"({100 * in_set / total:.4f}%)"
    )
    print("=== Done ===")


if __name__ == "__main__":
    main()
