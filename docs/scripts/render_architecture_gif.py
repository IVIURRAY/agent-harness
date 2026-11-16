"""Render docs/architecture.gif: the harness as an animated, hand-drawn wireframe.

Three lanes: the run (task -> loop <-> provider <-> model), the action path (approval gate ->
sandbox -> tools) and observability (events -> terminal UI + JSONL trace). Packets travel the
arrows in the order a real run produces them: a read-only call, an approved write, a path that
escapes the sandbox, then the final answer.

    uv run --group docs python docs/scripts/render_architecture_gif.py [--still 40]
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from sketch import (
    AMBER,
    AMBER_SOFT,
    BLUE,
    BLUE_SOFT,
    F_BOX,
    F_LANE,
    F_SUB,
    F_TITLE,
    GREEN,
    GREEN_SOFT,
    INK,
    MUTED,
    PINK,
    PINK_SOFT,
    RED,
    RED_SOFT,
    S,
    WHITE,
    along,
    caption_pill,
    dot,
    label_pill,
    paper,
    save_gif,
    sketch_arrow,
    sketch_box,
    text,
)

W, H = 1200, 660
SLOW = 1.6


def T(n: float) -> int:
    return round(n * SLOW)


@dataclass
class Box:
    key: str
    title: str
    sub: str
    x: int
    y: int
    w: int = 232
    h: int = 78

    @property
    def c(self):
        return (self.x + self.w / 2, self.y + self.h / 2)

    def right(self, dy=0):
        return (self.x + self.w, self.y + self.h / 2 + dy)

    def left(self, dy=0):
        return (self.x, self.y + self.h / 2 + dy)

    def top(self, dx=0):
        return (self.x + self.w / 2 + dx, self.y)

    def bottom(self, dx=0):
        return (self.x + self.w / 2 + dx, self.y + self.h)


COLS = [70, 352, 634, 916]
ROW1, ROW2, ROW3 = 138, 330, 528

B = {b.key: b for b in [
    Box("task", "Task + system prompt", "\"organise these files\"", COLS[0], ROW1),
    Box("loop", "Agent loop", "step budget · compaction", COLS[1], ROW1),
    Box("provider", "Provider", "Anthropic · OpenRouter · fake", COLS[2], ROW1),
    Box("llm", "LLM", "Claude / free model", COLS[3], ROW1),
    Box("guard", "Guardrails", "schema · repeat guard · 8k cap", COLS[0], ROW2),
    Box("approve", "Approval gate", "you: y / n / all / reason", COLS[1], ROW2),
    Box("sandbox", "Sandbox", "paths jailed to workspace", COLS[2], ROW2),
    Box("tools", "Filesystem tools", "list · read · mkdir · move", COLS[3], ROW2),
    Box("events", "Event bus",
        "run · status · tool_call · approval · tool_result · compaction · finished",
        COLS[0], ROW3, w=COLS[2] + 232 - COLS[0]),
    Box("subs", "Subscribers", "terminal UI · runs/*.jsonl", COLS[3], ROW3),
]}

ARROWS = [  # (from, to, dashed)
    (B["task"].right(), B["loop"].left(), False),
    (B["loop"].right(-12), B["provider"].left(-12), False),
    (B["provider"].left(12), B["loop"].right(12), False),
    (B["provider"].right(-12), B["llm"].left(-12), False),
    (B["llm"].left(12), B["provider"].right(12), False),
    (B["loop"].bottom(-20), B["approve"].top(-20), False),
    (B["approve"].top(20), B["loop"].bottom(20), False),
    (B["approve"].right(-12), B["sandbox"].left(-12), False),
    (B["sandbox"].left(12), B["approve"].right(12), False),
    (B["sandbox"].right(-12), B["tools"].left(-12), False),
    (B["tools"].left(12), B["sandbox"].right(12), False),
    (B["guard"].right(), B["approve"].left(), False),
    (B["guard"].bottom(), (B["guard"].c[0], ROW3), True),
    (B["approve"].bottom(), (B["approve"].c[0], ROW3), True),
    (B["sandbox"].bottom(), (B["sandbox"].c[0], ROW3), True),
    (B["events"].right(), B["subs"].left(), True),
]

FILLS = {"task": BLUE_SOFT, "loop": BLUE_SOFT, "provider": BLUE_SOFT, "llm": BLUE_SOFT,
         "guard": AMBER_SOFT, "approve": AMBER_SOFT, "sandbox": GREEN_SOFT, "tools": GREEN_SOFT,
         "events": PINK_SOFT, "subs": PINK_SOFT}


@dataclass
class Packet:
    start: int
    end: int
    path: list
    color: tuple
    label: str | None = None
    label_at: tuple | None = None
    size: int = 7
    linger: int = 16


def _mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


ABOVE_ROW1 = ROW1 - 26
BELOW_ROW1 = ROW1 + 78 + 34
BELOW_ROW2 = ROW2 + 78 + 30
ABOVE_ROW3 = ROW3 - 26


def script():
    pk: list[Packet] = []

    def move(start, end, a, b, color, label=None, at=None, **kw):
        pk.append(Packet(T(start), T(end), [a, b], color, label, label_at=at, **kw))

    def down_tools(start, color, label, back_label, blocked=False):
        """loop -> gate -> sandbox -> tools and back; `blocked` stops at the sandbox."""
        t = start
        move(t, t + 7, B["loop"].bottom(-20), B["approve"].top(-20), color, label,
             (B["approve"].c[0] - 40, BELOW_ROW1))
        move(t + 7, t + 14, B["approve"].right(-12), B["sandbox"].left(-12), color)
        if blocked:
            move(t + 16, t + 23, B["sandbox"].left(12), B["approve"].right(12), RED, back_label,
                 (B["sandbox"].c[0] - 40, BELOW_ROW2), linger=22)
            move(t + 23, t + 30, B["approve"].top(20), B["loop"].bottom(20), RED)
            return t + 30
        move(t + 14, t + 21, B["sandbox"].right(-12), B["tools"].left(-12), color)
        move(t + 22, t + 29, B["tools"].left(12), B["sandbox"].right(12), GREEN, back_label,
             (B["tools"].c[0], BELOW_ROW2))
        move(t + 29, t + 35, B["sandbox"].left(12), B["approve"].right(12), GREEN)
        move(t + 35, t + 41, B["approve"].top(20), B["loop"].bottom(20), GREEN)
        return t + 41

    def model_call(start, label):
        t = start
        move(t, t + 7, B["loop"].right(-12), B["provider"].left(-12), BLUE)
        move(t + 7, t + 14, B["provider"].right(-12), B["llm"].left(-12), BLUE)
        move(t + 15, t + 22, B["llm"].left(12), B["provider"].right(12), AMBER, label,
             (B["provider"].c[0] + 140, BELOW_ROW1))
        move(t + 22, t + 29, B["provider"].left(12), B["loop"].right(12), AMBER)
        return t + 29

    move(2, 11, B["task"].right(), B["loop"].left(), BLUE, "Organise the files", (B["task"].c[0] + 140, ABOVE_ROW1))
    t = model_call(12, "tool_use: list_files")
    t = down_tools(t + 1, GREEN, "list_files(.)", "10 files")  # read-only: gate passes it straight through
    t = model_call(t + 2, "tool_use: move_file")
    write_start = t + 1
    t = down_tools(write_start, AMBER, "move invoice -> finance/", "moved")
    t = model_call(t + 2, "tool_use: read_file")
    escape_start = t + 1
    t = down_tools(escape_start, AMBER, "read ../../.ssh/id_rsa", "blocked: escapes sandbox", blocked=True)
    t = model_call(t + 2, "no tool calls: done")
    end = t
    move(end + 1, end + 10, B["loop"].left(12), B["task"].right(12), BLUE, "Done: 4 folders",
         (B["task"].c[0] + 140, ABOVE_ROW1))

    # Telemetry dots fall from the action lane into the event bus as work happens.
    for f in range(14, end, 9):
        src = ("approve", "sandbox", "guard")[f % 3]
        x = B[src].c[0] + random.Random(f).uniform(-30, 30)
        pk.append(Packet(T(f), T(f + 7), [(x, B[src].y + B[src].h), (x, ROW3 + 22)], PINK, size=5))
    move(end - 4, end + 6, B["events"].right(), B["subs"].left(), PINK, "status line + trace",
         (B["subs"].c[0], ABOVE_ROW3))

    lit = {
        "task": [0, 12, end, end + 25], "loop": [8, end + 12], "provider": [12, end],
        "llm": [18, end], "guard": [42, 50, write_start, write_start + 6],
        "approve": [write_start, write_start + 14, escape_start, escape_start + 8],
        "sandbox": [42, 60, escape_start + 6, escape_start + 24], "tools": [50, 70, write_start + 14, write_start + 30],
        "events": [14, end + 6], "subs": [end - 2, end + 25],
    }
    lit = {k: [T(n) for n in v] for k, v in lit.items()}
    steps = [
        (0, 41, "You give it a task; the loop calls the model with the full history", BLUE),
        (41, write_start - 1, "Read-only tool: no approval needed, still runs inside the sandbox", GREEN),
        (write_start - 1, escape_start - 1, "Writes need you: the gate shows the move and waits for y / n", AMBER),
        (escape_start - 1, end - 6, "A path outside the workspace is refused in code, not by the prompt", RED),
        (end - 6, end + 25, "No tool calls = done. Every step was an event: status line + JSONL trace", PINK),
    ]
    captions = [(T(a), T(b), f"Step {i} of {len(steps)}  ·  {txt}", c) for i, (a, b, txt, c) in enumerate(steps, 1)]
    gate_flags = [(T(write_start + 6), T(write_start + 16), "approve? y", AMBER),
                  (T(escape_start + 13), T(escape_start + 26), "denied", RED)]
    return pk, lit, captions, gate_flags, T(end + 25)


def _lit(windows, f):
    return any(a <= f <= b for a, b in zip(windows[::2], windows[1::2]))


def render_frame(f, packets, lit, captions, gate_flags) -> Image.Image:
    img, d = paper(W, H)
    rng = random.Random(1000 + (f // 3) % 4)  # cartoon "boil": re-jitter every 3 frames

    text(d, (70, 46), "agent-harness", F_TITLE, INK, anchor="lm")
    text(d, (70, 78), "the LLM decides what to do next · the harness decides what's allowed", F_SUB, MUTED, anchor="lm")
    for label, y in (("THE RUN", ROW1 - 20), ("ACTION PATH (every tool call)", ROW2 - 20),
                     ("OBSERVABILITY", ROW3 - 20)):
        text(d, (70, y), label, F_LANE, MUTED, anchor="lm")

    for p, q, dashed in ARROWS:
        sketch_arrow(d, p, q, rng, MUTED, dashed=dashed)

    for key, b in B.items():
        on = _lit(lit.get(key, []), f)
        edge = BLUE if key == "loop" else INK
        fill = FILLS[key] if on else WHITE
        if key == "sandbox" and any(a <= f <= e for a, e, _, c in gate_flags if c == RED):
            fill, edge = RED_SOFT, RED
        sketch_box(d, b.x, b.y, b.w, b.h, rng, color=edge, fill=fill)
        text(d, (b.c[0], b.y + 30), b.title, F_BOX, INK)
        text(d, (b.c[0], b.y + 55), b.sub, F_SUB, MUTED)

    # Loop doodle: spins while a run is in progress.
    ax, ay = B["loop"].x + 26, B["loop"].y - 14
    spin = (f * 12) % 360 if _lit(lit["loop"], f) else 0
    d.arc([(ax - 12) * S, (ay - 12) * S, (ax + 12) * S, (ay + 12) * S], start=spin + 30, end=spin + 320,
          fill=BLUE, width=int(2.5 * S))
    tip = math.radians(spin + 320)
    tx, ty = ax + 12 * math.cos(tip), ay + 12 * math.sin(tip)
    d.polygon([((tx + 5 * math.cos(tip + 2.2)) * S, (ty + 5 * math.sin(tip + 2.2)) * S),
               ((tx + 7 * math.cos(tip + 1.97)) * S, ty * S), (tx * S, (ty - 6) * S)], fill=BLUE)

    for p in packets:
        if p.start <= f <= p.end:
            x, y = along(p.path, (f - p.start) / max(1, p.end - p.start))
            dot(d, x, y, p.color, p.size)
    for p in packets:
        if p.label and p.label_at and p.start <= f <= p.end + p.linger:
            label_pill(d, *p.label_at, p.label, p.color, anchor_pt=_mid(p.path[0], p.path[-1]))
    for a, e, s, color in gate_flags:
        if a <= f <= e:
            b = B["approve"] if color == AMBER else B["sandbox"]
            label_pill(d, b.x + b.w - 30, b.y - 4, s, color)
    for a, e, cap, color in captions:
        if a <= f <= e:
            caption_pill(d, W / 2, H - 28, cap, color)
    return img.resize((W, H), Image.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "architecture.gif"))
    ap.add_argument("--still", help="comma-separated frame numbers to save as PNGs instead")
    args = ap.parse_args()
    packets, lit, captions, flags, n = script()
    if args.still:
        for f in map(int, args.still.split(",")):
            out = Path(args.out).with_name(f"architecture-{f}.png")
            render_frame(f, packets, lit, captions, flags).save(out)
            print(f"wrote {out}")
        return
    save_gif([render_frame(f, packets, lit, captions, flags) for f in range(n)], Path(args.out))


if __name__ == "__main__":
    main()
