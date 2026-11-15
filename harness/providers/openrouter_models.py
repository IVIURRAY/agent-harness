"""Discover OpenRouter's free models that support tool calling.

The free roster changes often, so nothing is hard-coded - we ask the API at run time.

    python -m harness.providers.openrouter_models
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

MODELS_URL = "https://openrouter.ai/api/v1/models"


def is_free(model: dict[str, Any]) -> bool:
    pricing = model.get("pricing") or {}
    return model.get("id", "").endswith(":free") or (
        str(pricing.get("prompt")) in ("0", "0.0") and str(pricing.get("completion")) in ("0", "0.0")
    )


def supports_tools(model: dict[str, Any]) -> bool:
    return "tools" in (model.get("supported_parameters") or [])


def free_tool_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Free + tool-capable, largest context first (a rough proxy for 'serious' models)."""
    picked = [m for m in models if is_free(m) and supports_tools(m) and not m.get("id", "").startswith("openrouter/")]
    return sorted(picked, key=lambda m: (m.get("context_length") or 0, m.get("created") or 0), reverse=True)


def fetch_models(timeout: float = 20.0) -> list[dict[str, Any]]:
    request = urllib.request.Request(MODELS_URL, headers={"User-Agent": "agent-harness"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # raises on HTTP errors
        return json.load(response)["data"]


def main() -> None:
    models = free_tool_models(fetch_models())
    if not models:
        print("No free tool-capable models listed right now.")
        return
    print(f"{len(models)} free models with tool calling (largest context first):\n")
    for m in models:
        print(f"  {m['id']:<60} ctx {m.get('context_length') or '?':>8}")
    print("\nPin one with HARNESS_MODEL=<id> in ~/.env")


if __name__ == "__main__":
    main()
