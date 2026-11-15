"""Anthropic (Claude) provider using the official SDK.

Harness-relevant API behaviour encoded here:
  * The transcript is append-only. We append response.content unchanged (thinking blocks,
    fallback blocks and all). Recent Claude models bind thinking blocks to the exact
    conversation prefix; editing earlier turns gets the request rejected.
  * system prompt and tool list are frozen for the session (also keeps the prompt cache warm).
  * Compaction is the "simple" shape: replace the *whole* history with a summary, so no old
    thinking block is ever replayed against a different prefix.
  * Forced tool_choice is not supported on current models - we use auto + strict schemas.
  * Server-side refusal fallbacks are on by default ("default" routing).
"""

from __future__ import annotations

from typing import Any

import anthropic

from ..tools.base import Tool
from .base import (
    STOP_END,
    STOP_MAX_TOKENS,
    STOP_PAUSE,
    STOP_REFUSAL,
    STOP_TOOL_USE,
    ModelTurn,
    ProviderError,
    ToolCall,
    ToolOutcome,
    Usage,
)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
_STOP_MAP = {
    "tool_use": STOP_TOOL_USE,
    "end_turn": STOP_END,
    "stop_sequence": STOP_END,
    "max_tokens": STOP_MAX_TOKENS,
    "refusal": STOP_REFUSAL,
    "pause_turn": STOP_PAUSE,
}


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        effort: str = "medium",
        max_tokens: int = 16_000,
        fallbacks: bool = True,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks
        # Credentials are resolved by the SDK from the environment; the harness never reads
        # or forwards the key, so it can't leak into the prompt or traces.
        self.client = client or anthropic.Anthropic(max_retries=4)

    def start(self, system: str, tools: list[Tool]) -> "AnthropicSession":
        return AnthropicSession(self, system, tools)


class AnthropicSession:
    def __init__(self, provider: AnthropicProvider, system: str, tools: list[Tool]) -> None:
        self.p = provider
        self.system = system
        self.tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.json_schema(),
                "strict": True,
            }
            for t in tools
        ]
        self.messages: list[dict[str, Any]] = []

    # --- transcript -------------------------------------------------------------------
    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_tool_results(self, results: list[ToolOutcome]) -> None:
        # All results for one assistant turn go back in ONE user message.
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.call_id,
                        "content": r.content,
                        **({"is_error": True} if r.is_error else {}),
                    }
                    for r in results
                ],
            }
        )

    def reset_with(self, text: str) -> None:
        self.messages = [{"role": "user", "content": text}]

    # --- model calls ------------------------------------------------------------------
    def _create(self, messages: list[dict[str, Any]]):
        kwargs: dict[str, Any] = dict(
            model=self.p.model,
            max_tokens=self.p.max_tokens,
            system=self.system,
            tools=self.tools,
            messages=messages,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.p.effort},
            cache_control={"type": "ephemeral"},
        )
        if self.p.fallbacks:
            kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")
        try:
            return self.p.client.beta.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise ProviderError("Authentication failed - check ANTHROPIC_API_KEY in ~/.env") from exc
        except anthropic.BadRequestError as exc:
            raise ProviderError(f"Bad request: {exc.message}") from exc
        except anthropic.NotFoundError as exc:
            raise ProviderError(f"Model or endpoint not found ({self.p.model}): {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise ProviderError("Rate limited after retries", retryable=True) from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"API error {exc.status_code}: {exc.message}", retryable=exc.status_code >= 500) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError("Network error talking to the API", retryable=True) from exc

    def step(self) -> ModelTurn:
        response = self._create(self.messages)
        # Append-only: the reply goes into history exactly as received.
        self.messages.append({"role": "assistant", "content": response.content})

        turn = ModelTurn(stop=_STOP_MAP.get(response.stop_reason, STOP_END), model=response.model)
        texts, thoughts = [], []
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "thinking" and block.thinking:
                thoughts.append(block.thinking)
            elif block.type == "tool_use":
                turn.tool_calls.append(ToolCall(block.id, block.name, block.input))
            elif block.type == "fallback":
                turn.notes.append(f"fallback: {block.from_.model} declined, {block.to.model} continued")
        turn.text = "\n".join(texts).strip()
        turn.thinking = "\n".join(thoughts).strip()
        if response.stop_reason == "refusal" and getattr(response, "stop_details", None):
            turn.notes.append(f"refusal category: {response.stop_details.category}")

        u = response.usage
        turn.usage = Usage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        # A tool_use block with stop_reason max_tokens may be truncated - don't run it.
        if turn.stop == STOP_MAX_TOKENS:
            turn.tool_calls = []
        return turn

    def summarize(self, instruction: str) -> str:
        # Same system + tools (keeps the cached prefix), extra user turn, transcript untouched.
        response = self._create([*self.messages, {"role": "user", "content": instruction}])
        return "\n".join(b.text for b in response.content if b.type == "text").strip()
