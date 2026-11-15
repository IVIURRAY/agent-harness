"""Settings from ~/.env, ./.env and the environment. Secrets stay in the process env for the SDK."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Settings:
    model: str
    effort: str
    max_steps: int
    compact_at_tokens: int
    fallbacks: bool


def load_settings() -> Settings:
    load_dotenv(Path.home() / ".env")
    load_dotenv(Path.cwd() / ".env")  # project-local overrides
    return Settings(
        model=os.getenv("HARNESS_MODEL", "claude-sonnet-5-5"),
        effort=os.getenv("HARNESS_EFFORT", "medium"),
        max_steps=int(os.getenv("HARNESS_MAX_STEPS", "40")),
        compact_at_tokens=int(os.getenv("HARNESS_COMPACT_AT_TOKENS", "120000")),
        # Turn off on platforms without server-side fallbacks (Bedrock / Vertex / Foundry).
        fallbacks=os.getenv("HARNESS_FALLBACKS", "1") not in ("0", "false", "no"),
    )


def has_credentials() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
