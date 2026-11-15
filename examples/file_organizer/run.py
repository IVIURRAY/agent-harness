"""File-organising agent demo.

    python -m examples.file_organizer.run            # real model (needs ANTHROPIC_API_KEY in ~/.env)
    python -m examples.file_organizer.run --offline  # scripted model, no key needed
    python -m examples.file_organizer.run --yes      # auto-approve everything
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from harness import Agent, AgentConfig, EventBus
from harness.approvals import AutoApprove
from harness.config import has_credentials, load_settings
from harness.events import JsonlTrace
from harness.sandbox import Sandbox
from harness.tools import ToolRegistry, filesystem_tools
from harness.ui import ConsoleRenderer, TerminalApprovals

from .seed import seed

SYSTEM_PROMPT = """\
You are a careful file-organising assistant working inside a sandboxed workspace directory.
All paths are relative to the workspace root; you cannot reach anything outside it.

How to work:
- Start by listing the workspace, then read files whose purpose isn't obvious from the name.
- Decide on a small, sensible set of subdirectories (3-6), named in lowercase snake_case.
- Create each directory, then move each file into the right one.
- Directory creation and file moves need the user's approval. If the user declines an operation,
  respect it: don't retry it unchanged; adapt your plan, and take any reason they give into account.
- Before each batch of actions, say in one short sentence what you're about to do.
- Finish by listing the workspace again and giving a brief summary of the final layout."""

TASK = "Organise the files in this workspace into appropriate subdirectories."


def offline_script():
    from harness.providers.scripted import Script, ScriptedProvider

    m = lambda src, dst: ("move_file", {"source": src, "destination": dst})  # noqa: E731
    return ScriptedProvider([
        Script("Let me look at what's here.", [("list_files", {"path": "."})]),
        Script("A couple of names are ambiguous; reading them.",
               [("read_file", {"path": "standup_2026_11_10.txt"}), ("read_file", {"path": "notes_langgraph_checkpoints.txt"})]),
        Script("I'll create four folders.", [("create_directory", {"path": p}) for p in ("finance", "travel", "work", "recipes")]),
        Script("Moving files into place.", [
            m("invoice_march_acme.txt", "finance"), m("invoice_april_hosting.txt", "finance"),
            m("receipt_climbing_shoes.txt", "finance"), m("trip_ideas_portugal.txt", "travel"),
            m("packing_list_alps_hike.txt", "travel"), m("meeting_notes_q3_planning.txt", "work"),
            m("standup_2026_11_10.txt", "work"), m("notes_langgraph_checkpoints.txt", "work"),
            m("recipe_feijoada.txt", "recipes"), m("recipe_pao_de_queijo.txt", "recipes"),
        ]),
        Script("Checking the result.", [("list_files", {"path": "."})]),
        Script("Done: finance/ (3), travel/ (2), work/ (3), recipes/ (2)."),
    ])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="demo_files", help="workspace directory (re-seeded each run)")
    ap.add_argument("--offline", action="store_true", help="use the scripted provider (no API calls)")
    ap.add_argument("--yes", action="store_true", help="auto-approve every operation")
    ap.add_argument("--no-seed", action="store_true", help="don't reset the workspace first")
    ap.add_argument("--hide-thinking", action="store_true")
    args = ap.parse_args(argv)

    console = Console()
    settings = load_settings()
    workspace = Path(args.dir) if args.no_seed else seed(Path(args.dir))
    sandbox = Sandbox(workspace)

    if args.offline:
        provider = offline_script()
    else:
        if not has_credentials():
            console.print("[red]No ANTHROPIC_API_KEY found in ~/.env, ./.env or the environment.[/red] "
                          "Add one, or run with --offline.")
            return 2
        from harness.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(model=settings.model, effort=settings.effort, fallbacks=settings.fallbacks)

    bus = EventBus()
    bus.subscribe(ConsoleRenderer(console, show_thinking=not args.hide_thinking))
    trace = JsonlTrace(Path("runs"))
    bus.subscribe(trace)

    agent = Agent(
        provider=provider,
        system=SYSTEM_PROMPT,
        tools=ToolRegistry(filesystem_tools(sandbox)),
        approvals=AutoApprove() if args.yes else TerminalApprovals(console),
        bus=bus,
        config=AgentConfig(max_steps=settings.max_steps, compact_at_tokens=settings.compact_at_tokens),
    )
    try:
        result = agent.run(TASK)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        return 130
    console.print(f"[dim]trace: {trace.path}[/dim]")
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
