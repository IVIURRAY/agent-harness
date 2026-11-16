"""Tool abstraction. A tool is a schema the model sees plus a function the harness runs.

Two harness-owned properties live here rather than in the prompt:
  * requires_approval - whether a human must OK the call before it runs
  * describe()        - a human-readable preview of what the call will do (shown at approval time)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

MAX_RESULT_CHARS = 8_000  # cap what any single tool result can push into the context


class ToolError(Exception):
    """An expected failure the model should see and recover from (bad path, missing file...)."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]          # JSON Schema properties
    fn: Callable[..., str]
    required: list[str] = field(default_factory=list)
    requires_approval: bool = False
    untrusted_output: bool = False      # output comes from outside the harness (files, web...) -> screened
    describe: Callable[[dict[str, Any]], str] | None = None

    def json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": self.parameters,
            "required": self.required,
            "additionalProperties": False,
        }

    def preview(self, args: dict[str, Any]) -> str:
        if self.describe:
            try:
                return self.describe(args)
            except Exception:  # a broken preview must never block the approval prompt
                pass
        return f"{self.name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        # Sorted, fixed order: the tool list is part of the prompt-cache prefix, so keep it stable.
        self._tools = {t.name: t for t in sorted(tools, key=lambda t: t.name)}

    def __iter__(self):
        return iter(self._tools.values())

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def validate(self, tool: Tool, args: Any) -> str | None:
        """Minimal schema check. Returns an error message or None."""
        if not isinstance(args, dict):
            return "Tool input must be a JSON object."
        missing = [k for k in tool.required if k not in args]
        if missing:
            return f"Missing required argument(s): {', '.join(missing)}"
        unknown = [k for k in args if k not in tool.parameters]
        if unknown:
            return f"Unknown argument(s): {', '.join(unknown)}"
        for key, value in args.items():
            if tool.parameters[key].get("type") == "string" and not isinstance(value, str):
                return f"Argument {key!r} must be a string."
        return None


def truncate(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars by the harness]"
