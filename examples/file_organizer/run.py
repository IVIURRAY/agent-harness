"""File-organising agent demo.

    python -m examples.file_organizer.run            # real model (ANTHROPIC_API_KEY or OPENROUTER_API_KEY in ~/.env)
    python -m examples.file_organizer.run --provider openrouter   # free OpenRouter model, auto-selected
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
from harness.config import load_settings
from harness.factory import SetupError, build_provider
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
    """A scripted run for --offline mode: no API key, but the harness, approvals and filesystem
    operations are all real. It branches if you decline the receipt move, like a model would."""
    from harness.providers.scripted import Script, ScriptedProvider

    def m(src, dst):
        return ("move_file", {"source": src, "destination": dst})

    def receipt_declined(history):
        return any(call.name == "move_file" and "receipt" in call.args.get("source", "")
                   and outcome.is_error and "declined" in outcome.content
                   for call, outcome in history)

    return ScriptedProvider([
        Script("Let me look at what's here.", [("list_files", {"path": "."})]),
        Script("Two names are ambiguous, so I'll read them.",
               [("read_file", {"path": "standup_2026_11_10.txt"}),
                ("read_file", {"path": "notes_langgraph_checkpoints.txt"})]),
        Script("I'll group these into four folders: finance, travel, work and recipes.",
               [("create_directory", {"path": p}) for p in ("finance", "travel", "work", "recipes")]),
        Script("Moving each file into its folder.", [
            m("invoice_march_acme.txt", "finance"), m("invoice_april_hosting.txt", "finance"),
            m("receipt_climbing_shoes.txt", "finance"), m("trip_ideas_portugal.txt", "travel"),
            m("packing_list_alps_hike.txt", "travel"), m("meeting_notes_q3_planning.txt", "work"),
            m("standup_2026_11_10.txt", "work"), m("notes_langgraph_checkpoints.txt", "work"),
            m("recipe_feijoada.txt", "recipes"), m("recipe_pao_de_queijo.txt", "recipes"),
        ]),
        Script("You declined the receipt move, so I'll give receipts their own folder.",
               [("create_directory", {"path": "receipts"}), m("receipt_climbing_shoes.txt", "receipts")],
               when=receipt_declined),
        Script("Checking the final layout.", [("list_files", {"path": "."})]),
        Script("Done. Final layout: finance/ (invoices), receipts/ (receipts), travel/, work/ and recipes/.",
               when=receipt_declined),
        Script("Done. Final layout: finance/ (3), travel/ (2), work/ (3), recipes/ (2)."),
    ], latency=0.9)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="demo_files", help="workspace directory (re-seeded each run)")
    ap.add_argument("--offline", action="store_true", help="use the scripted provider (no API calls)")
    ap.add_argument("--yes", action="store_true", help="auto-approve every operation")
    ap.add_argument("--no-seed", action="store_true", help="don't reset the workspace first")
    ap.add_argument("--hide-thinking", action="store_true")
    ap.add_argument("--provider", choices=["anthropic", "openrouter"], help="default: HARNESS_PROVIDER or auto")
    args = ap.parse_args(argv)

    console = Console()
    settings = load_settings(args.provider)
    workspace = Path(args.dir) if args.no_seed else seed(Path(args.dir))
    sandbox = Sandbox(workspace)

    compact_at = settings.compact_at_tokens
    if args.offline:
        provider = offline_script()
    else:
        try:
            built = build_provider(settings)
        except SetupError as exc:
            console.print(f"[red]{exc}[/red] Add a key (see .env.example), or run with --offline.")
            return 2
        provider, compact_at = built.provider, built.compact_at_tokens
        for note in built.notes:
            console.print(f"[dim]{note}[/dim]")

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
        config=AgentConfig(max_steps=settings.max_steps, compact_at_tokens=compact_at),
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
