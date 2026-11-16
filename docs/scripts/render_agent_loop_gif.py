"""Render docs/agent-loop.gif: one run of the file organiser through the harness loop.

Three panels:

  left    what the model sees: system rules, tool schemas, the task
  centre  the loop: model call -> "tool calls in the reply?" -> harness (validate, approve,
          sandbox, append) -> back to the model, or END
  right   the conversation, growing one message at a time (append-only)

The point it makes: the *model* decides what to do next (by replying with or without tool
calls); the *harness* decides what is allowed, and a declined approval goes back to the model
as an error result it can adapt to. Mirrors Agent.run / Agent._execute in harness/loop.py.

    uv run --group docs python docs/scripts/render_agent_loop_gif.py [--still 120,300]
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path

from harness.config import DEFAULT_MAX_STEPS
from PIL import Image
from sketch import (
    AMBER,
    AMBER_SOFT,
    BLUE,
    BLUE_SOFT,
    F_BOX,
    F_SUB,
    F_TITLE,
    GREEN,
    GREEN_SOFT,
    INK,
    MUTED,
    RED,
    RED_SOFT,
    S,
    WHITE,
    along,
    caption_pill,
    dot,
    font,
    label_pill,
    paper,
    save_gif,
    sketch_arrow,
    sketch_box,
    sketch_diamond,
    sketch_path,
    text,
)

W, H = 1400, 700
F_SMALL = font(13)
F_PANEL = font(15, bold=True)
F_HEAD = font(14, bold=True)
F_MONO = font(14)
SLOW = 1.5


def T(n: float) -> int:
    return round(n * SLOW)


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def c(self):
        return (self.x + self.w / 2, self.y + self.h / 2)


# ------------------------------------------------------------------ layout
START = (610, 122)
MODEL = Rect(490, 150, 240, 80)
DIAMOND = (610, 320, 200, 110)  # cx, cy, w, h
END = Rect(392, 440, 190, 80)
HARNESS = Rect(690, 420, 280, 150)
HARNESS_LINES = ["validate args (schema)", "approve? (writes only)", "run inside the sandbox",
                 "append result to history"]
PANEL_R = Rect(1060, 128, 310, 480)

P_START_MODEL = [START, (MODEL.c[0], MODEL.y)]
P_MODEL_DIAMOND = [(MODEL.c[0], MODEL.y + MODEL.h), (DIAMOND[0], DIAMOND[1] - DIAMOND[3] / 2)]
P_YES = [(DIAMOND[0] + DIAMOND[2] / 2, DIAMOND[1]), (HARNESS.c[0], DIAMOND[1]), (HARNESS.c[0], HARNESS.y)]
P_NO = [(DIAMOND[0] - DIAMOND[2] / 2, DIAMOND[1]), (END.c[0], DIAMOND[1]), (END.c[0], END.y)]
P_BACK = [(HARNESS.x + HARNESS.w, HARNESS.c[1]), (1010, HARNESS.c[1]), (1010, MODEL.c[1]),
          (MODEL.x + MODEL.w, MODEL.c[1])]


# ------------------------------------------------------------------ conversation
@dataclass
class Msg:
    role: str
    lines: list[str]
    color: tuple
    fill: tuple | None = None

    @property
    def height(self) -> int:
        return 24 + 20 * len(self.lines) + 12


MSGS = [
    Msg("user", ["Organise the files in", "this workspace."], BLUE),
    Msg("assistant · ACT: tool call", ["list_files(path=\".\")"], AMBER),
    Msg("tool · OBSERVE: result", ["10 files: invoice_march,", "trip_ideas, recipe_feijoada..."], GREEN),
    Msg("assistant · ACT: tool calls", ["create_directory x4:", "finance travel work recipes"], AMBER),
    Msg("tool · OBSERVE: results", ["you approved 4 / 4", "created 4 folders"], GREEN),
    Msg("assistant · ACT: tool calls", ["move_file x10, e.g.", "receipt_shoes -> finance/"], AMBER),
    Msg("tool · OBSERVE: results", ["9 moved. receipt_shoes: ERROR", "user declined: \"keep receipts", "in their own folder\""], RED, RED_SOFT),
    Msg("assistant · ACT: adapts", ["create_directory(receipts)", "move_file(receipt_shoes", "  -> receipts/)"], AMBER),
    Msg("tool · OBSERVE: results", ["approved · moved"], GREEN),
    Msg("assistant · ANSWER (no tools)", ["Done: finance, travel, work,", "recipes, receipts"], BLUE),
]


# ------------------------------------------------------------------ timeline
@dataclass
class Seg:
    frames: int
    node: str | None = None          # model | diamond | harness | end
    path: list | None = None
    pcolor: tuple = BLUE
    msgs: int = 0                    # messages visible
    sub: int | None = None           # highlighted harness line
    flag: tuple | None = None        # (text, color) pill over the harness
    caption: tuple | None = None
    step: int = 0
    branch: str | None = None        # "yes" | "no" label emphasis
    extra: dict = field(default_factory=dict)


def build() -> list[Seg]:
    segs: list[Seg] = []
    cap = None
    state = {"msgs": 1, "step": 0}

    def add(frames, **kw):
        kw.setdefault("msgs", state["msgs"])
        kw.setdefault("step", state["step"])
        kw.setdefault("caption", cap)
        segs.append(Seg(T(frames), **kw))

    def model_turn(msg_after: int, color=AMBER):
        state["step"] += 1
        add(5, node=None, path=P_BACK if state["step"] > 1 else P_START_MODEL, pcolor=GREEN if state["step"] > 1 else BLUE)
        add(12, node="model")
        state["msgs"] = msg_after
        add(6, node="model")
        add(7, node="diamond", path=P_MODEL_DIAMOND, pcolor=color)

    def harness(approve=None, approve_color=AMBER, result_msgs=None):
        add(7, node="diamond", path=P_YES, pcolor=AMBER, branch="yes")
        add(6, node="harness", sub=0)
        if approve:
            add(14, node="harness", sub=1, flag=(approve, approve_color))
        add(6, node="harness", sub=2)
        state["msgs"] = result_msgs
        add(8, node="harness", sub=3)

    cap = ("The model sees: system rules + tool schemas + the whole conversation", BLUE)
    add(14, msgs=1)
    model_turn(2)
    cap = ("Read-only tool call: no approval, but still validated and sandboxed", GREEN)
    harness(result_msgs=3)
    cap = ("Reason + act again: results are appended, the model picks the next step", AMBER)
    model_turn(4)
    cap = ("Writes stop at the approval gate: you see the operation and answer", AMBER)
    harness(approve="you: y y y y", result_msgs=5)
    model_turn(6)
    cap = ("You decline one move, with a reason: it goes back as an error result", RED)
    harness(approve="you: r \"keep receipts in their own folder\"", approve_color=RED, result_msgs=7)
    cap = ("The model adapts to the reason instead of retrying the same call", AMBER)
    model_turn(8)
    harness(approve="you: y", result_msgs=9)
    cap = ("No tool calls in the reply: the loop ends with a summary", BLUE)
    model_turn(10, color=BLUE)
    add(7, node="diamond", path=P_NO, pcolor=BLUE, branch="no")
    add(30, node="end")
    return segs


def frames_from(segs):
    for s in segs:
        for i in range(s.frames):
            yield s, i / max(1, s.frames - 1)


# ------------------------------------------------------------------ drawing
def left_panel(d, rng):
    text(d, (36, 112), "WHAT THE MODEL SEES", F_PANEL, MUTED, anchor="lm")
    blocks = [
        (128, 160, "System prompt", GREEN, ["• List, then read unclear files", "• 3-6 snake_case folders",
                                             "• Moves need approval; a decline", "   means adapt, don't retry",
                                             "• Summarise the final layout"]),
        (304, 136, "Tools (JSON schemas)", AMBER, ["list_files · read_file", "create_directory *",
                                                   "delete_directory *  · move_file *", "* = needs your approval"]),
        (456, 78, "Task", BLUE, ["\"Organise the files in", "this workspace.\""]),
    ]
    for y, h, title, color, lines in blocks:
        sketch_box(d, 36, y, 300, h, rng, color=INK, fill=WHITE)
        text(d, (52, y + 22), title, F_HEAD, color, anchor="lm")
        for i, line in enumerate(lines):
            text(d, (52, y + 48 + i * 21), line, F_SMALL, INK, anchor="lm")
    text(d, (36, 558), "+ the conversation so far (right)", F_SMALL, MUTED, anchor="lm")
    text(d, (36, 580), "+ the API key? never: the SDK reads it", F_SMALL, MUTED, anchor="lm")


def centre(d, rng, seg: Seg, t: float, f: int):
    text(d, (392, 112), "THE HARNESS LOOP", F_PANEL, MUTED, anchor="lm")
    # START
    d.ellipse([(START[0] - 9) * S, (START[1] - 9) * S, (START[0] + 9) * S, (START[1] + 9) * S], fill=INK)
    text(d, (START[0] + 18, START[1]), "START", F_SMALL, MUTED, anchor="lm")
    sketch_arrow(d, (START[0], START[1] + 9), (MODEL.c[0], MODEL.y), rng, MUTED)

    on = seg.node
    # model node
    sketch_box(d, MODEL.x, MODEL.y, MODEL.w, MODEL.h, rng, color=BLUE,
               fill=BLUE_SOFT if on == "model" else WHITE)
    text(d, (MODEL.c[0], MODEL.y + 30), "model call", F_BOX, INK)
    text(d, (MODEL.c[0], MODEL.y + 56), "provider.step(history)", F_SUB, MUTED)
    if seg.step:
        label_pill(d, MODEL.x + 4, MODEL.y - 2, f"step {seg.step} / {DEFAULT_MAX_STEPS}", INK)
    text(d, (MODEL.x + MODEL.w + 16, MODEL.y + 8), "before each call:", F_SMALL, MUTED, anchor="lm")
    text(d, (MODEL.x + MODEL.w + 16, MODEL.y + 28), "compact if context > 60%", F_SMALL, MUTED, anchor="lm")

    # decision
    sketch_arrow(d, *P_MODEL_DIAMOND, rng, MUTED)
    cx, cy, dw, dh = DIAMOND
    sketch_diamond(d, cx, cy, dw, dh, rng, fill=AMBER_SOFT if on == "diamond" else (250, 244, 230))
    text(d, (cx, cy - 11), "tool calls", F_SUB, INK)
    text(d, (cx, cy + 11), "in the reply?", F_SUB, INK)
    sketch_path(d, P_NO, rng, MUTED)
    sketch_path(d, P_YES, rng, MUTED)
    text(d, (cx - dw / 2 - 26, cy - 14), "no", F_HEAD, BLUE if seg.branch == "no" else MUTED)
    text(d, (cx + dw / 2 + 26, cy - 14), "yes", F_HEAD, AMBER if seg.branch == "yes" else MUTED)

    # END
    sketch_box(d, END.x, END.y, END.w, END.h, rng, fill=BLUE_SOFT if on == "end" else WHITE)
    text(d, (END.c[0], END.y + 30), "END", F_BOX, INK)
    text(d, (END.c[0], END.y + 56), "final summary", F_SUB, MUTED)

    # harness node
    fill = GREEN_SOFT if on == "harness" else WHITE
    if on == "harness" and seg.flag and seg.flag[1] == RED and seg.sub == 1:
        fill = RED_SOFT
    sketch_box(d, HARNESS.x, HARNESS.y, HARNESS.w, HARNESS.h, rng, fill=fill)
    text(d, (HARNESS.c[0], HARNESS.y + 24), "harness: per tool call", F_BOX, INK)
    for i, line in enumerate(HARNESS_LINES):
        active = on == "harness" and seg.sub == i
        color = INK if active else MUTED
        y = HARNESS.y + 54 + i * 22
        if active:
            d.ellipse([(HARNESS.x + 18) * S, (y - 4) * S, (HARNESS.x + 26) * S, (y + 4) * S], fill=GREEN)
        text(d, (HARNESS.x + 34, y), line, F_HEAD if active else F_SUB, color, anchor="lm")
    sketch_path(d, P_BACK, rng, MUTED)
    text(d, (1018, (HARNESS.c[1] + MODEL.c[1]) / 2), "results", F_SMALL, MUTED, anchor="lm")
    text(d, (HARNESS.x, HARNESS.y + HARNESS.h + 18), "guardrails: step budget · repeat guard · 8k result cap",
         F_SMALL, MUTED, anchor="lm")

    if seg.flag and on == "harness":
        label_pill(d, HARNESS.c[0], HARNESS.y - 26, seg.flag[0], seg.flag[1])

    if seg.path:
        x, y = along(seg.path, t)
        dot(d, x, y, seg.pcolor)


def right_panel(d, rng, seg: Seg):
    text(d, (PANEL_R.x, 112), "CONVERSATION (append-only)", F_PANEL, MUTED, anchor="lm")
    sketch_box(d, PANEL_R.x, PANEL_R.y, PANEL_R.w, PANEL_R.h, rng, fill=WHITE)
    visible = MSGS[: seg.msgs]
    # Keep the newest messages in view; older ones scroll off the top.
    budget, shown = PANEL_R.h - 24, []
    for m in reversed(visible):
        if budget - m.height < 0:
            break
        budget -= m.height
        shown.insert(0, m)
    y = PANEL_R.y + 14
    if len(shown) < len(visible):
        text(d, (PANEL_R.x + 18, y + 2), f"... {len(visible) - len(shown)} earlier messages", F_SMALL, MUTED, anchor="lm")
        y += 22
    for i, m in enumerate(shown):
        newest = i == len(shown) - 1
        if m.fill or newest:
            d.rounded_rectangle([(PANEL_R.x + 8) * S, (y - 4) * S, (PANEL_R.x + PANEL_R.w - 8) * S,
                                 (y + m.height - 12) * S], radius=6 * S,
                                fill=m.fill or (252, 246, 228))
        text(d, (PANEL_R.x + 18, y + 10), m.role, F_HEAD, m.color, anchor="lm")
        for j, line in enumerate(m.lines):
            text(d, (PANEL_R.x + 18, y + 32 + j * 20), line, F_MONO, INK, anchor="lm")
        y += m.height


def render(seg: Seg, t: float, f: int) -> Image.Image:
    img, d = paper(W, H)
    rng = random.Random(2000 + (f // 3) % 4)
    text(d, (36, 46), "Inside the harness loop: reason, act, observe", F_TITLE, INK, anchor="lm")
    text(d, (36, 78), "the model chooses the next action · the harness validates, asks you, runs it in the sandbox, "
                      "and appends the result", F_SUB, MUTED, anchor="lm")
    left_panel(d, rng)
    centre(d, rng, seg, t, f)
    right_panel(d, rng, seg)
    if seg.caption:
        caption_pill(d, W / 2, H - 30, seg.caption[0], seg.caption[1])
    return img.resize((W, H), Image.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "agent-loop.gif"))
    ap.add_argument("--still", help="comma-separated frame numbers to save as PNGs instead")
    args = ap.parse_args()
    timeline = list(frames_from(build()))
    if args.still:
        for f in map(int, args.still.split(",")):
            out = Path(args.out).with_name(f"agent-loop-{f}.png")
            render(*timeline[min(f, len(timeline) - 1)], f).save(out)
            print(f"wrote {out}")
        return
    print(f"{len(timeline)} frames")
    save_gif([render(seg, t, f) for f, (seg, t) in enumerate(timeline)], Path(args.out))


if __name__ == "__main__":
    main()
