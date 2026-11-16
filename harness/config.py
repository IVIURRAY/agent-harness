"""Settings from ~/.env, ./.env and the environment. Secrets stay in the process env for the SDKs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


# The one place these defaults live. AgentConfig (loop.py) and .env both resolve to them.
DEFAULT_MAX_STEPS = 40                # model calls per run - the hard stop on runaway loops
DEFAULT_COMPACT_AT_TOKENS = 120_000   # compact the conversation when the context reaches this size


@dataclass
class Settings:
    provider: str                     # anthropic | openrouter
    model: str | None                 # None = provider default / auto-discover
    effort: str
    max_steps: int
    compact_at_tokens: int
    fallbacks: bool
    fallback_models: list[str] = field(default_factory=list)


def _env_list(name: str) -> list[str]:
    return [s.strip() for s in os.getenv(name, "").split(",") if s.strip()]


def load_settings(provider_override: str | None = None) -> Settings:
    load_dotenv(Path.home() / ".env")
    load_dotenv(Path.cwd() / ".env")  # project-local overrides
    provider = (provider_override or os.getenv("HARNESS_PROVIDER") or "auto").lower()
    if provider == "auto":
        provider = "openrouter" if os.getenv("OPENROUTER_API_KEY") and not os.getenv("ANTHROPIC_API_KEY") else "anthropic"
    return Settings(
        provider=provider,
        model=os.getenv("HARNESS_MODEL") or None,
        effort=os.getenv("HARNESS_EFFORT", "medium"),
        max_steps=int(os.getenv("HARNESS_MAX_STEPS", DEFAULT_MAX_STEPS)),
        compact_at_tokens=int(os.getenv("HARNESS_COMPACT_AT_TOKENS", DEFAULT_COMPACT_AT_TOKENS)),
        # Anthropic server-side refusal fallbacks. Turn off on Bedrock / Vertex / Foundry.
        fallbacks=os.getenv("HARNESS_FALLBACKS", "1") not in ("0", "false", "no"),
        fallback_models=_env_list("HARNESS_FALLBACK_MODELS"),
    )


def credential_for(provider: str) -> str | None:
    if provider == "openrouter":
        return os.getenv("OPENROUTER_API_KEY")
    return os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
