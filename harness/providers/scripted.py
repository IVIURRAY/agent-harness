"""A deterministic fake provider. Lets you test the loop, approvals, sandbox and compaction
without an API key, and makes harness behaviour reproducible in CI."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..tools.base import Tool
from .base import STOP_END, STOP_TOOL_USE, ModelTurn, ToolCall, ToolOutcome, Usage


@dataclass
class Script:
    """One scripted model turn: optional text plus zero or more (tool_name, args) calls.

    `when` lets a script branch on what happened so far (e.g. only run if the user declined
    something): it gets every (ToolCall, ToolOutcome) pair so far; the turn is skipped if it
    returns False.
    """
    text: str = ""
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    context_tokens: int = 1_000
    when: Callable[[list[tuple[ToolCall, ToolOutcome]]], bool] | None = None


class ScriptedProvider:
    name = "scripted"
    model = "scripted"

    def __init__(self, turns: list[Script], summary: str = "Summary of progress so far.", latency: float = 0.0) -> None:
        self.turns = turns
        self.summary = summary
        self.latency = latency  # seconds of fake "thinking" per call, for demos
        self.session: ScriptedSession | None = None

    def start(self, system: str, tools: list[Tool]) -> "ScriptedSession":
        self.session = ScriptedSession(self)
        return self.session


class ScriptedSession:
    def __init__(self, provider: ScriptedProvider) -> None:
        self.p = provider
        self.transcript: list[tuple[str, Any]] = []
        self.results: list[ToolOutcome] = []
        self.calls: dict[str, ToolCall] = {}
        self._turns = iter(provider.turns)
        self._ids = itertools.count(1)

    def add_user(self, text: str) -> None:
        self.transcript.append(("user", text))

    def add_tool_results(self, results: list[ToolOutcome]) -> None:
        self.results.extend(results)
        self.transcript.append(("tool_results", results))

    def reset_with(self, text: str) -> None:
        self.transcript = [("user", text)]

    def summarize(self, instruction: str) -> str:
        return self.p.summary

    def step(self) -> ModelTurn:
        if self.p.latency:
            time.sleep(self.p.latency)
        history = [(self.calls[r.call_id], r) for r in self.results if r.call_id in self.calls]
        script = next((s for s in self._turns if s.when is None or s.when(history)), Script(text="Done."))
        calls = [ToolCall(f"call_{next(self._ids)}", name, args) for name, args in script.calls]
        self.calls.update({c.id: c for c in calls})
        turn = ModelTurn(
            stop=STOP_TOOL_USE if calls else STOP_END,
            text=script.text,
            tool_calls=calls,
            usage=Usage(input_tokens=script.context_tokens),
            model="scripted",
        )
        self.transcript.append(("assistant", turn))
        return turn
