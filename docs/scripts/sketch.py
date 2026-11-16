"""Hand-drawn sketch primitives shared by the README GIF renderers.

Adapted from the renderers in IVIURRAY/agentic-research-assistant so the READMEs share one look.
Uses a handwriting-style font when available (macOS: Chalkboard SE / Noteworthy; Linux: Comic Neue).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

S = 2  # supersampling factor for smooth sketch lines
PAPER = (250, 248, 242)
GRID = (232, 236, 242)
INK = (38, 42, 52)
MUTED = (120, 126, 138)
BLUE = (47, 91, 211)
BLUE_SOFT = (222, 231, 252)
GREEN = (31, 140, 85)
GREEN_SOFT = (220, 242, 228)
AMBER = (214, 140, 20)
AMBER_SOFT = (252, 238, 210)
PINK = (204, 64, 120)
PINK_SOFT = (250, 226, 236)
RED = (200, 50, 50)
RED_SOFT = (252, 225, 225)
WHITE = (255, 255, 255)

# (regular, bold, ttc index for bold) - first one that exists wins
FONT_CANDIDATES = [
    ("/System/Library/Fonts/Supplemental/ChalkboardSE.ttc", None, 1),
    ("/System/Library/Fonts/Noteworthy.ttc", None, 1),
    ("/usr/share/fonts/opentype/comic-neue/ComicNeue-Regular.otf",
     "/usr/share/fonts/opentype/comic-neue/ComicNeue-Bold.otf", 0),
    ("/usr/share/fonts/truetype/comic-neue/ComicNeue-Regular.ttf",
     "/usr/share/fonts/truetype/comic-neue/ComicNeue-Bold.ttf", 0),
]
MONO_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for regular, bold_path, bold_index in FONT_CANDIDATES:
        path = bold_path if (bold and bold_path) else regular
        if not Path(path).exists():
            continue
        try:
            return ImageFont.truetype(path, size * S, index=bold_index if (bold and not bold_path) else 0)
        except OSError:
            try:
                return ImageFont.truetype(path, size * S)
            except OSError:
                continue
    return ImageFont.load_default(size * S)


def mono(size: int) -> ImageFont.FreeTypeFont:
    for path in MONO_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size * S)
    return ImageFont.load_default(size * S)


F_TITLE = font(26, bold=True)
F_BOX = font(19, bold=True)
F_SUB = font(14)
F_LANE = font(14, bold=True)
F_TAG = font(15, bold=True)
F_CAP = font(17, bold=True)


def jitter_line(d: ImageDraw.ImageDraw, p, q, rng: random.Random, color, width=2.2, wobble=2.5):
    """A slightly bowed, slightly overshooting stroke - drawn twice like a pencil sketch."""
    (x1, y1), (x2, y2) = p, q
    length = math.hypot(x2 - x1, y2 - y1) or 1
    nx, ny = -(y2 - y1) / length, (x2 - x1) / length
    for _ in range(2):
        bow = rng.uniform(-wobble, wobble) * min(1.0, length / 120)
        ox1, oy1 = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
        ox2, oy2 = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
        pts = []
        for i in range(24):
            t = i / 23
            bx = x1 + ox1 + (x2 + ox2 - x1 - ox1) * t + nx * bow * math.sin(math.pi * t)
            by = y1 + oy1 + (y2 + oy2 - y1 - oy1) * t + ny * bow * math.sin(math.pi * t)
            pts.append((bx * S, by * S))
        d.line(pts, fill=color, width=int(width * S), joint="curve")


def sketch_box(d, x, y, w, h, rng, color=INK, fill=None):
    if fill:
        d.rounded_rectangle([(x + 2) * S, (y + 2) * S, (x + w - 2) * S, (y + h - 2) * S],
                            radius=10 * S, fill=fill)
    r = 10
    corners = [(x + r, y), (x + w - r, y), (x + w, y + r), (x + w, y + h - r),
               (x + w - r, y + h), (x + r, y + h), (x, y + h - r), (x, y + r)]
    for i in range(0, 8, 2):
        jitter_line(d, corners[i], corners[i + 1], rng, color)
        jitter_line(d, corners[i + 1], corners[(i + 2) % 8], rng, color, wobble=0.6)


def sketch_diamond(d, cx, cy, w, h, rng, color=INK, fill=None):
    pts = [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)]
    if fill:
        d.polygon([(x * S, y * S) for x, y in pts], fill=fill)
    for a, b in zip(pts, pts[1:] + pts[:1]):
        jitter_line(d, a, b, rng, color, wobble=1.0)


def sketch_arrow(d, p, q, rng, color=MUTED, dashed=False, head=True):
    if dashed:
        (x1, y1), (x2, y2) = p, q
        n = max(2, int(math.hypot(x2 - x1, y2 - y1) / 14))
        for i in range(0, n, 2):
            a = (x1 + (x2 - x1) * i / n, y1 + (y2 - y1) * i / n)
            b = (x1 + (x2 - x1) * (i + 1) / n, y1 + (y2 - y1) * (i + 1) / n)
            jitter_line(d, a, b, rng, color, width=2, wobble=0.3)
    else:
        jitter_line(d, p, q, rng, color, width=2, wobble=1.2)
    if head:
        ang = math.atan2(q[1] - p[1], q[0] - p[0])
        for side in (-1, 1):
            a2 = ang + math.pi + side * 0.45
            tip = (q[0] + 11 * math.cos(a2), q[1] + 11 * math.sin(a2))
            jitter_line(d, q, tip, rng, color, width=2, wobble=0.2)


def sketch_path(d, pts, rng, color=MUTED, dashed=False):
    """Polyline with an arrowhead on the last segment only."""
    for i, (a, b) in enumerate(zip(pts, pts[1:])):
        sketch_arrow(d, a, b, rng, color, dashed=dashed, head=(i == len(pts) - 2))


def text(d, xy, s, f, color=INK, anchor="mm"):
    d.text((xy[0] * S, xy[1] * S), s, font=f, fill=color, anchor=anchor)


def label_pill(d, cx, cy, s, color, anchor_pt=None, f=None):
    """Solid, high-contrast data label. Optional leader line to the arrow it describes."""
    f = f or F_TAG
    l, t, r, b = d.textbbox((cx * S, cy * S), s, font=f, anchor="mm")
    pad_x, pad_y = 10 * S, 6 * S
    box = [l - pad_x, t - pad_y, r + pad_x, b + pad_y]
    if anchor_pt:
        ax, ay = anchor_pt
        ey = box[3] if ay * S > box[3] else box[1]
        d.line([(cx * S, ey), (ax * S, ay * S)], fill=color, width=2 * S)
        d.ellipse([(ax - 3) * S, (ay - 3) * S, (ax + 3) * S, (ay + 3) * S], fill=color)
    d.rounded_rectangle(box, radius=10 * S, fill=color, outline=INK, width=S)
    d.text((cx * S, cy * S), s, font=f, fill=WHITE, anchor="mm")


def caption_pill(d, cx, cy, s, color):
    l, t, r, b = d.textbbox((cx * S, cy * S), s, font=F_CAP, anchor="mm")
    pad_x, pad_y = 14 * S, 8 * S
    d.rounded_rectangle([l - pad_x, t - pad_y, r + pad_x, b + pad_y], radius=12 * S,
                        fill=color, outline=INK, width=S)
    d.text((cx * S, cy * S), s, font=F_CAP, fill=WHITE, anchor="mm")


def paper(w: int, h: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (w * S, h * S), PAPER)
    d = ImageDraw.Draw(img)
    for gx in range(0, w, 24):
        d.line([(gx * S, 0), (gx * S, h * S)], fill=GRID, width=S)
    for gy in range(0, h, 24):
        d.line([(0, gy * S), (w * S, gy * S)], fill=GRID, width=S)
    return img, d


def ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def along(path, t):
    segs = list(zip(path, path[1:]))
    lengths = [math.dist(a, b) for a, b in segs]
    target = ease(t) * sum(lengths)
    for (a, b), L in zip(segs, lengths):
        if target <= L:
            f = target / L if L else 0
            return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
        target -= L
    return path[-1]


def dot(d, x, y, color, r=7):
    d.ellipse([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S], fill=color, outline=INK, width=S)


def save_gif(frames: list[Image.Image], out: Path, frame_ms: int = 70, hold_ms: int = 2500,
             seed_colors=(GREEN, BLUE, AMBER, PINK, RED, INK, WHITE), colors: int = 160,
             durations: list[int] | None = None) -> None:
    """Shared adaptive palette sampled across the timeline keeps files small and colours stable."""
    w, h = frames[0].size
    picks = [frames[i] for i in range(min(5, len(frames) - 1), len(frames), max(1, len(frames) // 24))]
    src = Image.new("RGB", (w // 2, (h // 2) * len(picks) + 40))
    for i, fr in enumerate(picks):
        src.paste(fr.resize((w // 2, h // 2)), (0, 40 + i * (h // 2)))
    sd = ImageDraw.Draw(src)
    for i, c in enumerate(seed_colors):
        sd.rectangle([i * 60, 0, i * 60 + 55, 40], fill=c)
    pal = src.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    gif = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    if durations is None:
        durations = [frame_ms] * len(gif)
        durations[-1] = hold_ms
    out.parent.mkdir(parents=True, exist_ok=True)
    gif[0].save(out, save_all=True, append_images=gif[1:], duration=durations, loop=0,
                optimize=True, disposal=1)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(gif)} frames)")
