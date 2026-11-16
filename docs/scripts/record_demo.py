"""Record docs/demo.gif: the real terminal UI, driven end to end.

Runs `python -m examples.file_organizer.run --offline` in a pseudo-terminal, answers the
approval prompts like a person would (including one decline with a reason), and renders the
captured terminal output to a GIF with pyte (a terminal emulator) + Pillow. No asciinema/agg/vhs.

--offline means a scripted model, but everything else is real: the harness loop, the approval
prompts, the sandbox and the file moves.

    uv run --group docs python docs/scripts/record_demo.py
"""

from __future__ import annotations

import argparse
import fcntl
import os
import pty
import select
import struct
import sys
import termios
import time
from pathlib import Path

import pyte
from PIL import Image, ImageDraw, ImageFont
from sketch import save_gif

ROOT = Path(__file__).resolve().parents[2]
COLS, ROWS = 96, 30

# Answers typed at each approval prompt, in order. 4 folders, 2 moves, decline the receipt
# with a reason, then "yes to all" for the rest.
ANSWERS = ["y", "y", "y", "y", "y", "y", ("r", "keep receipts in their own folder"), "a"]

# ------------------------------------------------------------------ capture


def capture(cmd: list[str]) -> list[tuple[float, bytes]]:
    pid, fd = pty.fork()
    if pid == 0:  # child
        os.chdir(ROOT)
        env = {**os.environ, "TERM": "xterm-256color", "COLUMNS": str(COLS), "LINES": str(ROWS),
               "COLORTERM": "truecolor", "PYTHONUNBUFFERED": "1"}
        os.execvpe(cmd[0], cmd, env)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))

    events: list[tuple[float, bytes]] = []
    t0 = time.monotonic()
    answers = list(ANSWERS)
    tail = b""

    def send(text: str) -> None:
        for ch in text:
            os.write(fd, ch.encode())
            time.sleep(0.07)  # human-ish typing (echo arrives as output)
        time.sleep(0.25)
        os.write(fd, b"\r")

    while True:
        if time.monotonic() - t0 > 180:  # a missed prompt must never hang the recorder
            os.kill(pid, 9)
            raise SystemExit("recording timed out - did the prompt text change?")
        r, _, _ = select.select([fd], [], [], 0.05)
        if r:
            try:
                data = os.read(fd, 65536)
            except OSError:
                break
            if not data:
                break
            events.append((time.monotonic() - t0, data))
            tail = (tail + data)[-400:]
            if b"y/n/a/r" in tail and answers:
                tail = b""
                time.sleep(1.1)  # "reading" the proposed operation
                ans = answers.pop(0)
                if isinstance(ans, tuple):
                    send(ans[0])
                    time.sleep(0.6)
                    send(ans[1])
                else:
                    send(ans)
        elif os.waitpid(pid, os.WNOHANG)[0]:
            return events
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
    return events


# ------------------------------------------------------------------ render

BG = (13, 17, 23)
FG = (225, 228, 232)
CHROME = (30, 35, 43)
PALETTE = {
    "black": (60, 64, 72), "red": (248, 81, 73), "green": (63, 185, 80), "brown": (210, 153, 34),
    "yellow": (210, 153, 34), "blue": (88, 166, 255), "magenta": (188, 140, 255),
    "cyan": (57, 197, 207), "white": (225, 228, 232),
    "brightblack": (139, 148, 158), "brightred": (255, 123, 114), "brightgreen": (86, 211, 100),
    "brightyellow": (227, 179, 65), "brightblue": (121, 192, 255), "brightmagenta": (210, 168, 255),
    "brightcyan": (86, 212, 221), "brightwhite": (255, 255, 255),
}
FONT_SIZE = 15
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # has the Braille spinner glyphs
EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
EMOJI_CHARS = set("✋⚙")


def _font(path: str, size: int):
    return ImageFont.truetype(path, size) if Path(path).exists() else ImageFont.load_default(size)


F_REG, F_BOLD, F_SANS = _font(MONO, FONT_SIZE), _font(MONO_BOLD, FONT_SIZE), _font(SANS, FONT_SIZE)
CW = round(F_REG.getlength("M"))
CH = round(FONT_SIZE * 1.28)
PAD, BAR = 18, 34
WIDTH, HEIGHT = COLS * CW + 2 * PAD, ROWS * CH + 2 * PAD + BAR
_emoji_cache: dict[str, Image.Image | None] = {}


def _emoji(ch: str) -> Image.Image | None:
    if ch in _emoji_cache:
        return _emoji_cache[ch]
    img = None
    if Path(EMOJI).exists():
        try:
            f = ImageFont.truetype(EMOJI, 109)  # CBDT bitmap font: only this size exists
            box = f.getbbox(ch)
            canvas = Image.new("RGBA", (box[2] + 4, box[3] + 4), (0, 0, 0, 0))
            ImageDraw.Draw(canvas).text((0, 0), ch, font=f, embedded_color=True)
            canvas = canvas.crop(canvas.getbbox())
            side = CH - 2
            img = canvas.resize((round(side * canvas.width / canvas.height), side), Image.LANCZOS)
        except OSError:
            img = None
    _emoji_cache[ch] = img
    return img


def _color(c: str, default):
    if c == "default":
        return default
    if c in PALETTE:
        return PALETTE[c]
    if len(c) == 6:
        try:
            return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            pass
    return default


def is_emoji(s: str) -> bool:
    return any(ord(c) >= 0x1F000 or c in EMOJI_CHARS for c in s)


def render(screen: pyte.Screen, title: str) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, WIDTH, BAR], fill=CHROME)
    for i, c in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
        d.ellipse([16 + i * 22, 11, 28 + i * 22, 23], fill=c)
    d.text((WIDTH / 2, BAR / 2), title, font=F_REG, fill=(150, 157, 167), anchor="mm")
    for y in range(screen.lines):
        row = screen.buffer[y]
        for x in range(screen.columns):
            cell = row[x]
            ch = cell.data.replace("️", "")
            fg = _color(cell.fg, FG)
            bg = _color(cell.bg, BG)
            if cell.reverse:
                fg, bg = bg, fg
            px, py = PAD + x * CW, BAR + PAD + y * CH
            if bg != BG:
                d.rectangle([px, py, px + CW, py + CH], fill=bg)
            if not ch or ch == " ":
                continue
            if is_emoji(ch):
                e = _emoji(ch)
                if e is not None:
                    img.paste(e, (px, py + 1), e)
                continue
            f = F_SANS if any(0x2800 <= ord(c) <= 0x28FF for c in ch) else (F_BOLD if cell.bold else F_REG)
            d.text((px, py + CH / 2), ch, font=f, fill=fg, anchor="lm")
    return img


def to_frames(events, fps=12.5, max_idle=1.4, hold=3.0):
    """Sample the terminal at a fixed rate; merge identical frames; squash long idle gaps."""
    screen = pyte.Screen(COLS, ROWS)
    stream = pyte.ByteStream(screen)
    step = 1 / fps
    frames, durations = [], []
    i, t, last_change = 0, 0.0, 0.0
    end = events[-1][0] + 0.1
    while t <= end:
        changed = False
        while i < len(events) and events[i][0] <= t:
            stream.feed(events[i][1])
            i += 1
            changed = True
        if changed or not frames:
            frames.append(render(screen, "agent-harness · file organiser (--offline)"))
            durations.append(step)
            last_change = t
        else:
            if t - last_change < max_idle:
                durations[-1] += step
        t += step
    durations[-1] = hold
    return frames, durations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/demo.gif"))
    ap.add_argument("--speed", type=float, default=1.0, help=">1 plays back faster")
    args = ap.parse_args()
    cmd = [sys.executable, "-m", "examples.file_organizer.run", "--offline"]
    events = capture(cmd)
    print(f"captured {len(events)} chunks over {events[-1][0]:.1f}s")
    frames, durations = to_frames(events)
    ms = [max(40, round(d * 1000 / args.speed)) for d in durations]
    out = Path(args.out)
    save_gif(frames, out, frame_ms=0, hold_ms=ms[-1], colors=128,
             seed_colors=tuple(PALETTE.values())[:12] + (BG, FG, CHROME), durations=ms)


if __name__ == "__main__":
    main()
