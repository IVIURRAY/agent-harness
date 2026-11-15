"""OpenAI-compatible Chat Completions provider: OpenRouter, OpenAI, Ollama, vLLM, LM Studio...

Same harness contract as the Anthropic provider (native, append-only transcript), but the
wire format differs in ways the harness has to absorb:
  * tool arguments arrive as a JSON *string* - may be malformed on weaker/free models
  * one `role: tool` message per tool result (no is_error flag - we prefix "ERROR:")
  * finish_reason is unreliable on some models; we trust the presence of tool_calls instead
  * OpenRouter can return HTTP 200 with an `error` object and no choices
"""

from __future__ import annotations

import json
from typing import Any

import openai

from ..tools.base import Tool
from .base import (
    STOP_END,
    STOP_MAX_TOKENS,
    STOP_REFUSAL,
    STOP_TOOL_USE,
    ModelTurn,
    ProviderError,
    ToolCall,
    ToolOutcome,
    Usage,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenAICompatProvider:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = OPENROUTER_BASE_URL,
        name: str = "openrouter",
        max_tokens: int = 8_000,
        fallback_models: list[str] | None = None,
        client: Any = None,
    ) -> None:
        self.name = name
        self.model = model
        self.max_tokens = max_tokens
        self.fallback_models = fallback_models or []
        self.is_openrouter = "openrouter.ai" in base_url
        # Free models are rate-limited hard; the SDK retries 429/5xx with backoff (honours retry-after).
        self.client = client or openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=5,
            default_headers={"X-Title": "agent-harness"} if self.is_openrouter else None,
        )

    def start(self, system: str, tools: list[Tool]) -> "OpenAICompatSession":
        return OpenAICompatSession(self, system, tools)


class OpenAICompatSession:
    def __init__(self, provider: OpenAICompatProvider, system: str, tools: list[Tool]) -> None:
        self.p = provider
        self.tools = [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.json_schema()},
            }
            for t in tools
        ]
        self.system_message = {"role": "system", "content": system}
        self.messages: list[dict[str, Any]] = [self.system_message]

    # --- transcript -------------------------------------------------------------------
    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_tool_results(self, results: list[ToolOutcome]) -> None:
        for r in results:
            content = f"ERROR: {r.content}" if r.is_error else r.content
            self.messages.append({"role": "tool", "tool_call_id": r.call_id, "content": content})

    def reset_with(self, text: str) -> None:
        self.messages = [self.system_message, {"role": "user", "content": text}]

    # --- model calls ------------------------------------------------------------------
    def _create(self, messages: list[dict[str, Any]]):
        extra_body: dict[str, Any] = {}
        if self.p.is_openrouter:
            # Only route to upstream providers that actually support every param we send (i.e. tools).
            extra_body["provider"] = {"require_parameters": True}
            if self.p.fallback_models:
                # OpenRouter tries these in order if the primary is down / rate-limited.
                extra_body["models"] = [self.p.model, *self.p.fallback_models]
        try:
            response = self.p.client.chat.completions.create(
                model=self.p.model,
                messages=messages,
                tools=self.tools,
                max_tokens=self.p.max_tokens,
                extra_body=extra_body or None,
            )
        except openai.AuthenticationError as exc:
            raise ProviderError("Authentication failed - check OPENROUTER_API_KEY in ~/.env") from exc
        except openai.NotFoundError as exc:
            raise ProviderError(
                f"Model not found or no endpoint matches ({self.p.model}). On OpenRouter, free models can "
                f"also need the free-model data policy enabled in your privacy settings. {exc.message}"
            ) from exc
        except openai.RateLimitError as exc:
            raise ProviderError("Rate limited after retries (free-model limits are low)", retryable=True) from exc
        except openai.BadRequestError as exc:
            raise ProviderError(f"Bad request: {exc.message}") from exc
        except openai.APIStatusError as exc:
            raise ProviderError(f"API error {exc.status_code}: {exc.message}", retryable=exc.status_code >= 500) from exc
        except openai.APIConnectionError as exc:
            raise ProviderError("Network error talking to the API", retryable=True) from exc

        error = getattr(response, "error", None) or (response.model_extra or {}).get("error")
        if error or not response.choices:
            raise ProviderError(f"Provider returned no choices: {error or 'empty response'}", retryable=True)
        return response

    def step(self) -> ModelTurn:
        response = self._create(self.messages)
        choice = response.choices[0]
        msg = choice.message

        # Append-only, and keep any extra fields the router returned (e.g. reasoning_details,
        # which some reasoning models need echoed back to keep their chain intact).
        stored = msg.model_dump(exclude_none=True)
        stored.setdefault("content", "")
        self.messages.append(stored)

        turn = ModelTurn(stop=STOP_END, model=getattr(response, "model", "") or self.p.model)
        turn.text = (msg.content or "").strip()
        reasoning = getattr(msg, "reasoning", None) or (msg.model_extra or {}).get("reasoning")
        if isinstance(reasoning, str):
            turn.thinking = reasoning.strip()

        for tc in msg.tool_calls or []:
            raw = tc.function.arguments or "{}"
            try:
                args: Any = json.loads(raw)
            except json.JSONDecodeError:
                args = raw  # loop's validator turns a non-object into an error result the model sees
            turn.tool_calls.append(ToolCall(tc.id, tc.function.name, args))

        finish = choice.finish_reason
        if turn.tool_calls and finish != "length":
            turn.stop = STOP_TOOL_USE
        elif finish == "length":
            turn.stop, turn.tool_calls = STOP_MAX_TOKENS, []   # truncated calls are never run
        elif finish == "content_filter":
            turn.stop = STOP_REFUSAL
            turn.notes.append("content_filter")
        if turn.model and turn.model != self.p.model:
            turn.notes.append(f"served by {turn.model}")

        u = response.usage
        if u is not None:
            details = getattr(u, "prompt_tokens_details", None)
            cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
            turn.usage = Usage(
                input_tokens=(u.prompt_tokens or 0) - cached,
                output_tokens=u.completion_tokens or 0,
                cache_read_tokens=cached,
            )
        return turn

    def summarize(self, instruction: str) -> str:
        response = self._create([*self.messages, {"role": "user", "content": instruction}])
        return (response.choices[0].message.content or "").strip()
