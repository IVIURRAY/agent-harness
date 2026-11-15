"""Build a provider from Settings. Keeps provider-specific setup out of apps."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings, credential_for


class SetupError(Exception):
    pass


@dataclass
class Built:
    provider: object
    compact_at_tokens: int
    notes: list[str] = field(default_factory=list)


def build_provider(settings: Settings) -> Built:
    key = credential_for(settings.provider)
    env_name = "OPENROUTER_API_KEY" if settings.provider == "openrouter" else "ANTHROPIC_API_KEY"
    if not key:
        raise SetupError(f"No {env_name} found in ~/.env, ./.env or the environment.")

    if settings.provider == "anthropic":
        from .providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(model=settings.model or "claude-sonnet-5-5", effort=settings.effort,
                                     fallbacks=settings.fallbacks)
        return Built(provider, settings.compact_at_tokens)

    if settings.provider == "openrouter":
        from .providers.openai_compat import OpenAICompatProvider

        notes: list[str] = []
        model, fallbacks, compact_at = settings.model, settings.fallback_models, settings.compact_at_tokens
        if model is None:
            from .providers.openrouter_models import fetch_models, free_tool_models

            try:
                candidates = free_tool_models(fetch_models())
            except Exception as exc:  # network, JSON, HTTP
                raise SetupError(f"Couldn't list OpenRouter models ({exc}). Set HARNESS_MODEL explicitly.") from exc
            if not candidates:
                raise SetupError("OpenRouter lists no free tool-capable models right now. Set HARNESS_MODEL.")
            model = candidates[0]["id"]
            fallbacks = fallbacks or [m["id"] for m in candidates[1:3]]
            ctx = candidates[0].get("context_length") or 0
            if ctx:
                # Small free-model windows: compact well before the limit.
                compact_at = min(compact_at, int(ctx * 0.6))
            notes.append(f"auto-selected free model {model} (ctx {ctx:,}); pin with HARNESS_MODEL")
            if fallbacks:
                notes.append(f"OpenRouter fallbacks: {', '.join(fallbacks)}")
        provider = OpenAICompatProvider(model=model, api_key=key, fallback_models=fallbacks)
        return Built(provider, compact_at, notes)

    raise SetupError(f"Unknown provider {settings.provider!r} (use anthropic or openrouter)")
