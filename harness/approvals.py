"""Approval gates: the human-in-the-loop seam.

The loop asks a gate before running any tool marked requires_approval. Gates are swappable:
interactive terminal prompt for the demo, auto-approve/deny for tests and CI, and in a real
deployment perhaps a Slack button or a policy engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .tools.base import Tool


@dataclass
class Decision:
    approved: bool
    reason: str = ""
    approve_rest: bool = False   # "yes to all" - skip prompting for the rest of the run


class ApprovalGate(Protocol):
    def decide(self, tool: Tool, args: dict[str, Any], preview: str) -> Decision: ...


class AutoApprove:
    def decide(self, tool, args, preview) -> Decision:
        return Decision(True, "auto-approved")


class DenyAll:
    def decide(self, tool, args, preview) -> Decision:
        return Decision(False, "denied by policy")


class ScriptedApprovals:
    """Approve/deny in a fixed order - for tests."""

    def __init__(self, answers: list[bool]) -> None:
        self.answers = list(answers)
        self.seen: list[str] = []

    def decide(self, tool, args, preview) -> Decision:
        self.seen.append(preview)
        ok = self.answers.pop(0) if self.answers else True
        return Decision(ok, "" if ok else "user said no")
