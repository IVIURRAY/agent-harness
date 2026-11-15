"""Terminal rendering. A plain EventBus subscriber - swap it for a web UI without touching the loop."""

from __future__ import annotations

import json

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from . import events as ev
from .approvals import Decision
from .events import Event

PHASE_ICON = {"thinking": "🧠", "acting": "🛠 ", "awaiting approval": "✋", "executing": "⚙️ ", "compacting": "🗜 "}


class ConsoleRenderer:
    def __init__(self, console: Console | None = None, show_thinking: bool = True) -> None:
        self.c = console or Console()
        self.show_thinking = show_thinking
        self._status = None
        self._step = "?"

    # The status line: one live spinner describing what the loop is doing *right now*.
    def _set_status(self, text: str) -> None:
        if self._status is None:
            self._status = self.c.status(text, spinner="dots")
            self._status.start()
        else:
            self._status.update(text)

    def _clear_status(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def __call__(self, e: Event) -> None:
        d = e.data
        if e.kind == ev.RUN_STARTED:
            self.c.print(Panel(Text(d["task"]), title=f"Task  ·  {d['provider']}:{d['model']}", border_style="cyan"))
            self.c.print(f"[dim]tools: {', '.join(d['tools'])}[/dim]")
        elif e.kind == ev.STATUS:
            if "step" in d:
                self._step = f"{d['step']}/{d['max_steps']}"
            icon = PHASE_ICON.get(d["phase"], "•")
            text = f"[bold]step {self._step}[/bold]  {icon} {d['phase']}  [dim]{d.get('detail', '')}[/dim]"
            if d["phase"] == "awaiting approval":
                self._clear_status()  # the prompt needs the terminal
            else:
                self._set_status(text)
        elif e.kind == ev.MODEL_RESPONSE:
            self._clear_status()
            u = d["usage"]
            cached = f", {u['cache_read_tokens']:,} cached" if u["cache_read_tokens"] else ""
            self.c.print(f"[dim]── step {self._step} · stop={d['stop']} · ctx≈{d['context_tokens']:,} tok"
                         f"{cached} · out {u['output_tokens']:,}[/dim]")
            for note in d.get("notes", []):
                self.c.print(f"[yellow]   note: {note}[/yellow]")
        elif e.kind == ev.THINKING and self.show_thinking:
            self.c.print(Text("   💭 " + _first_lines(d["text"], 3), style="dim italic"))
        elif e.kind == ev.ASSISTANT_TEXT:
            self.c.print(Text("   🤖 " + d["text"], style="white"))
        elif e.kind == ev.TOOL_CALL:
            self._clear_status()
            self.c.print(f"   [cyan]→ {d['name']}[/cyan] [dim]{_short_json(d['args'])}[/dim]")
        elif e.kind == ev.TOOL_RESULT:
            self._clear_status()
            style = "green" if d["ok"] else "red"
            mark = "✓" if d["ok"] else "✗"
            self.c.print(f"     [{style}]{mark}[/{style}] [dim]{_first_lines(d['output'], 4)}[/dim]")
        elif e.kind == ev.APPROVAL_DECIDED:
            verdict = "[green]approved[/green]" if d["approved"] else "[red]declined[/red]"
            self.c.print(f"     {verdict} [dim]{d.get('reason', '')}[/dim]")
        elif e.kind == ev.COMPACTION:
            self._clear_status()
            self.c.print(Panel(_first_lines(d["summary"], 12), title=f"🗜  compacted (was ≈{d['before_tokens']:,} tokens)",
                               border_style="magenta"))
        elif e.kind == ev.GUARDRAIL:
            self.c.print(f"   [yellow]⚠ guardrail: {d['reason']}[/yellow]")
        elif e.kind == ev.RUN_FINISHED:
            self._clear_status()
            colour = "green" if d["status"] == "completed" else "red"
            body = d["final_text"] or d["error"] or "(no final message)"
            stats = (f"steps {d['steps']} · tool calls {d['tool_calls']} · tokens in {d['input_tokens']:,} / "
                     f"out {d['output_tokens']:,} · compactions {d['compactions']}")
            self.c.print(Panel(Text(body), title=f"Run {d['status']}", subtitle=stats, border_style=colour))


class TerminalApprovals:
    """Shows the proposed operation and asks y / n / a(ll) / q(uit-with-reason)."""

    def __init__(self, console: Console | None = None) -> None:
        self.c = console or Console()

    def decide(self, tool, args, preview) -> Decision:
        self.c.print(Panel(Text(preview, style="bold"), title=f"Approve {tool.name}?", border_style="yellow",
                           subtitle="y = yes · n = no · a = yes to all · r = no, with a reason"))
        answer = Prompt.ask("  decision", choices=["y", "n", "a", "r"], default="y", console=self.c)
        if answer == "y":
            return Decision(True)
        if answer == "a":
            return Decision(True, "approved all remaining", approve_rest=True)
        if answer == "r":
            return Decision(False, Prompt.ask("  reason (sent to the agent)", console=self.c))
        return Decision(False)


def _first_lines(text: str, n: int) -> str:
    lines = text.strip().splitlines()
    out = "\n       ".join(lines[:n])
    return out + (f"\n       … (+{len(lines) - n} lines)" if len(lines) > n else "")


def _short_json(obj, limit: int = 140) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "…"
