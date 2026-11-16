"""Guardrails: the checks the harness runs around the model, so the model doesn't have to be trusted.

Each check either passes, or "trips" and returns a Trip: a reason for the trace/UI and a message
for the model. A tripped check never raises: the model gets the message as a tool error or a
user turn, and can recover.

Lives here (one place to read, tune and test):
  check_call()   per tool call, before anything runs
                   * unknown tool
                   * argument schema (required / unknown / types)
                   * repeat guard: the same call N times in a row is blocked
  cap_result()   per tool result: truncate so one tool can't flood the context
  check_reply()  per model reply, before the loop decides the run is over
                   * cut off at max_tokens -> ask the model to continue
                   * asked a question in text instead of calling a tool -> nudge once to act

Deliberately elsewhere, because they are the enforcement point itself:
  harness/sandbox.py        path jail (traversal, absolute paths, symlink escapes)
  harness/approvals.py      human approval for writes (wired in Agent._execute)
  harness/tools/fs.py       no overwrites, no deleting non-empty dirs or the root
  harness/loop.py           step budget (AgentConfig.max_steps); tool crashes -> "Internal tool error"
  harness/providers/*       truncated tool calls are dropped, never run

Not implemented yet: content guardrails (e.g. prompt-injection screening of file contents before
they enter the context). They would slot in next to cap_result().
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .providers.base import STOP_MAX_TOKENS, ModelTurn, ToolCall
from .tools.base import Tool, ToolRegistry, truncate

CUT_OFF_MESSAGE = "Your previous reply was cut off by the output limit. Continue, more concisely."

ACT_DONT_ASK_MESSAGE = (
    "You asked a question instead of acting. Nobody can answer in text during this run: when you "
    "call a tool that needs approval, the harness asks the user itself. So call the tools you need "
    "now. If the task is already complete, reply with your final summary and no question."
)

# A reply "asks instead of acting" if it ends with a question, or uses a permission phrase.
_PERMISSION = re.compile(
    r"\b(shall i|should i|would you like( me)?|do you want( me)?|want me to|may i|can i proceed|"
    r"let me know (if|whether)|please confirm|confirm (that|if|whether)|before i proceed|"
    r"ready to proceed|proceed\?)",
    re.IGNORECASE,
)


@dataclass
class Trip:
    """A guardrail fired. `reason` goes to the trace/UI; `message` goes to the model."""
    reason: str
    message: str


@dataclass
class GuardrailConfig:
    max_identical_calls: int = 3      # same tool + same args this many times in a row -> blocked
    max_result_chars: int = 8_000     # cap on what one tool result can add to the context
    nudge_text_questions: bool = True
    max_nudges: int = 1               # per run: bounded, so a false positive costs one extra turn


class Guardrails:
    def __init__(self, tools: ToolRegistry, config: GuardrailConfig | None = None) -> None:
        self.tools = tools
        self.config = config or GuardrailConfig()
        self.reset()

    def reset(self) -> None:
        """Clear per-run state. Called at the start of every run."""
        self._last_call_sig = ""
        self._repeat_count = 0
        self._nudges = 0

    # ---------------------------------------------------------------- per tool call
    def check_call(self, call: ToolCall) -> tuple[Tool | None, Trip | None]:
        """Returns (tool, None) if the call may proceed, else (tool_or_None, Trip)."""
        tool = self.tools.get(call.name)
        if tool is None:
            return None, Trip(f"unknown tool {call.name!r}", f"Unknown tool {call.name!r}.")
        if problem := self.tools.validate(tool, call.args):
            return tool, Trip(f"invalid arguments for {call.name}: {problem}", f"Invalid arguments: {problem}")

        sig = f"{call.name}:{json.dumps(call.args, sort_keys=True)}"
        self._repeat_count = self._repeat_count + 1 if sig == self._last_call_sig else 1
        self._last_call_sig = sig
        if self._repeat_count > self.config.max_identical_calls:
            return tool, Trip(
                f"blocked repeated call {call.name}",
                "Blocked by harness: this exact call was repeated too many times. Try something different.",
            )
        return tool, None

    # ---------------------------------------------------------------- per tool result
    def cap_result(self, output: str) -> str:
        return truncate(output, self.config.max_result_chars)

    # ---------------------------------------------------------------- per model reply
    def check_reply(self, turn: ModelTurn) -> Trip | None:
        """Checked when a reply has no tool calls to run. A Trip means: append the message as a
        user turn and keep looping, rather than treating the run as finished."""
        if turn.tool_calls:
            return None
        if turn.stop == STOP_MAX_TOKENS:
            return Trip("reply hit max_tokens; asking the model to continue", CUT_OFF_MESSAGE)
        if (self.config.nudge_text_questions and self._nudges < self.config.max_nudges
                and asks_instead_of_acting(turn.text)):
            self._nudges += 1
            return Trip("model asked a question in text instead of calling a tool; nudging it to act",
                        ACT_DONT_ASK_MESSAGE)
        return None


def asks_instead_of_acting(text: str) -> bool:
    """Heuristic: does the reply end on a question, or ask for permission?"""
    tail = text.strip()[-300:]
    if not tail:
        return False
    return tail.endswith("?") or bool(_PERMISSION.search(tail))


__all__ = ["Guardrails", "GuardrailConfig", "Trip", "asks_instead_of_acting",
           "CUT_OFF_MESSAGE", "ACT_DONT_ASK_MESSAGE"]

