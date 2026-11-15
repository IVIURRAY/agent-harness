"""Event bus: the harness's single observability seam.

Every interesting thing the loop does is emitted as an Event. The terminal UI, the JSONL
trace writer and (later) an OpenTelemetry exporter are all just subscribers. The loop
itself never prints anything.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Event kinds (kept as plain strings so traces stay greppable)
RUN_STARTED = "run_started"
STATUS = "status"                    # succinct "what is the loop doing now"
MODEL_REQUEST = "model_request"
MODEL_RESPONSE = "model_response"
ASSISTANT_TEXT = "assistant_text"
THINKING = "thinking"                # summarised reasoning, if the provider returns it
TOOL_CALL = "tool_call"
APPROVAL_REQUESTED = "approval_requested"
APPROVAL_DECIDED = "approval_decided"
TOOL_RESULT = "tool_result"
COMPACTION = "compaction"
GUARDRAIL = "guardrail"
RUN_FINISHED = "run_finished"


@dataclass
class Event:
    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


Handler = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._handlers: list[Handler] = []

    def subscribe(self, handler: Handler) -> None:
        self._handlers.append(handler)

    def emit(self, kind: str, **data: Any) -> None:
        event = Event(kind, data)
        for handler in self._handlers:
            handler(event)


class JsonlTrace:
    """Writes every event to runs/<timestamp>.jsonl - cheap, replayable, diffable."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / time.strftime("%Y%m%d-%H%M%S.jsonl")
        self._fh = self.path.open("a", encoding="utf-8")

    def __call__(self, event: Event) -> None:
        self._fh.write(json.dumps({"ts": event.ts, "kind": event.kind, **event.data}, default=str) + "\n")
        self._fh.flush()
