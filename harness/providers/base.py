"""Provider boundary.

Design choice: each provider keeps the transcript in its *native* message format and the
loop only sees normalised turns (ModelTurn / ToolCall / ToolOutcome). Converting history
to a neutral format and back on every call is lossy - it drops provider-specific blocks
(thinking signatures, fallback markers, cache hints) that must be replayed byte-for-byte.
So the neutral layer is the *interface*, not the *storage*.

Adding a provider (OpenAI, Bedrock, a local model) = implement ProviderSession.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..tools.base import Tool


@dataclass
class ToolCall:
    id: str
    name: str
    args: Any


@dataclass
class ToolOutcome:
    call_id: str
    content: str
    is_error: bool = False


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def context_tokens(self) -> int:
        """Approximate size of the conversation as of this turn (what compaction watches)."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens + self.output_tokens


# Normalised stop reasons
STOP_TOOL_USE = "tool_use"
STOP_END = "end_turn"
STOP_MAX_TOKENS = "max_tokens"
STOP_REFUSAL = "refusal"
STOP_PAUSE = "pause"          # provider wants us to re-send to continue


@dataclass
class ModelTurn:
    stop: str
    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    notes: list[str] = field(default_factory=list)   # e.g. "fallback: X declined, Y continued"


class ProviderError(Exception):
    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ProviderSession(Protocol):
    def add_user(self, text: str) -> None: ...
    def step(self) -> ModelTurn:
        """Call the model on the current transcript and append its reply to the transcript."""
        ...
    def add_tool_results(self, results: list[ToolOutcome]) -> None: ...
    def summarize(self, instruction: str) -> str:
        """Side request over the current transcript. Must NOT mutate the transcript."""
        ...
    def reset_with(self, text: str) -> None:
        """Replace the whole transcript with a single user message (simple compaction)."""
        ...


class Provider(Protocol):
    name: str
    model: str
    def start(self, system: str, tools: list[Tool]) -> ProviderSession: ...
