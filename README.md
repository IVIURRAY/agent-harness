# agent-harness

A small agent harness built from scratch in about 1,000 lines of Python (plus tests and the demo). It's meant for learning, and the
code is written to be read.

> **Harness** = the code around an LLM that supplies context, tools and guardrails, and runs the
> "call model → run tools → feed results back" loop until the task is done.

The demo is the file-organising agent from *The Batch* (13 Nov 2026). It runs with the tools jailed to
one directory, asks for approval before every mutation, and shows tool activity and a live status line.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# No key needed: scripted model, real harness, real filesystem ops
python -m examples.file_organizer.run --offline

# Real model: put ANTHROPIC_API_KEY in ~/.env (see .env.example)
python -m examples.file_organizer.run

pytest
```

Flags: `--yes` (auto-approve), `--hide-thinking`, `--dir PATH`, `--no-seed`.
Set `HARNESS_COMPACT_AT_TOKENS=4000` to watch compaction happen during the demo.
Every run writes a JSONL trace to `runs/`.

## Mental model

```
             ┌──────────────────────────── Agent.run(task) ─────────────────────────────┐
 task ──────►│ transcript ──► [compact if ctx > N] ──► provider.step() ──► ModelTurn    │
             │      ▲                                                       │           │
             │      │                                          tool calls? ─┤ no ──► done
             │      │                                                       ▼ yes        │
             │  tool results ◄── execute ◄── approval gate ◄── validate + guardrails     │
             │  (one message)   (sandbox)    (if mutating)    (schema, repeats, limits)  │
             └──────────────┬───────────────────────────────────────────────────────────┘
                            │ every transition
                            ▼
                     EventBus ──► ConsoleRenderer (status line, tool activity)
                              ──► JsonlTrace (runs/*.jsonl)  ──► (OTel exporter, later)
```

The LLM decides *what* to do next. The harness decides *what is allowed*, *what the model
sees*, and *when to stop*.

## Layout

| File | Role |
|---|---|
| `harness/loop.py` | The loop: step limit, tool dispatch, validation, approvals, repeat guard, compaction, events |
| `harness/providers/base.py` | Provider boundary: normalised `ModelTurn` / `ToolCall` / `ToolOutcome` |
| `harness/providers/anthropic_provider.py` | Claude via the official SDK |
| `harness/providers/scripted.py` | Deterministic fake model for tests and `--offline` |
| `harness/tools/` | `Tool` (schema + fn + `requires_approval` + human preview), sandboxed fs tools |
| `harness/sandbox.py` | Path jail (traversal, absolute paths, symlink escapes) |
| `harness/approvals.py` | Pluggable approval gates (auto, deny, scripted) |
| `harness/ui.py` | Terminal renderer and interactive approval prompt. Just an event subscriber |
| `harness/events.py` | Event bus and JSONL trace |
| `examples/file_organizer/` | The demo: seed data, system prompt, wiring |

## Design decisions worth defending (interview material)

**1. Enforce restrictions in code, not in the prompt.** The system prompt tells the model it is
sandboxed, but `Sandbox.resolve()` is what actually stops it. Every model-supplied path is resolved
(symlinks followed) and checked against the root. The model never gets raw filesystem access.
The tools also refuse to overwrite or delete non-empty dirs, so even an approved mistake is recoverable.

**2. The transcript is provider-native and append-only.** The loop talks to a neutral interface, but each
provider stores history in its own format. Converting to a common format and back on every call is
lossy. On current Claude models it also breaks things: thinking blocks are tied to the exact conversation
prefix they were produced in, so editing or rebuilding earlier turns gets the request rejected. The
same discipline keeps the prompt cache warm, so system prompt and tool list are frozen for the session.

**3. Compaction is "summarise everything and restart", not "summarise the old half".** When context
crosses a threshold, the harness makes a side request ("summarise progress, retain X, Y, Z"),
then replaces the whole transcript with `task + summary`. Keep-tail compaction (summary plus the last N
turns verbatim) looks nicer but replays thinking blocks against a prefix that no longer exists.
Compaction runs only between rounds, never with a tool call left unanswered.

**4. Approvals are a harness concern with a human-readable preview.** Tools declare
`requires_approval` and a `describe()` preview ("Move a.txt → finance/"), so the human approves
an *operation*, not raw JSON. A decline goes back to the model as an error result with the user's
reason, so the agent can change its plan instead of retrying.

**5. Every tool failure becomes a tool result, never a crash.** Unknown tool, bad args, sandbox
violation, user decline and tool exception all come back as `is_error` results the model can react
to. Internal exceptions are logged as guardrail events and the model only sees "Internal tool error".
Stack traces don't go into the context.

**6. Runaway protection.** These are hard limits: `max_steps` model calls, an 8k-character cap per tool
result, and blocking of the same tool with the same args more than 3 times in a row.

**7. Credentials never touch the context.** The SDK reads `ANTHROPIC_API_KEY` from the process
env. The harness never reads it, puts it in a prompt, or traces it.

**8. One observability seam.** The loop emits events and never prints. The UI, the JSONL trace and a
future OTel exporter (`step` → span, `tool_call`/`tool_result` → child spans) are all subscribers.

### Claude-specific choices (in `anthropic_provider.py`)
- `claude-sonnet-5-5` by default (as in the article), `effort=medium`. Override with `HARNESS_MODEL` / `HARNESS_EFFORT`.
- Adaptive thinking with `display: "summarized"`, so the UI can show a short "💭" line per step.
- `strict: true` tool schemas with `tool_choice` left on auto (current models reject forced tool choice).
- Server-side refusal fallbacks on (`fallbacks="default"`). Set `HARNESS_FALLBACKS=0` on Bedrock, Vertex or Foundry.
- Top-level automatic prompt caching. The UI shows cached tokens per step.
- `max_tokens` stops discard any (possibly truncated) tool calls and ask the model to continue.

## MVP vs later

**In the MVP (done):** loop, tools, sandbox, approvals, status and activity UI, compaction, guardrails,
traces, offline provider, tests.

**Worth adding next, in order of value:**
1. **Evals.** A handful of seeded workspaces plus a checker (every file moved? sensible groupings? no
   declined op retried?). Run with the scripted provider in CI and the real model nightly. This is
   what tells you a prompt or model change helped.
2. **A second provider** (OpenAI or a local model) behind `ProviderSession`. This proves the abstraction.
3. **OTel exporter** subscriber to Phoenix or MLflow.
4. **Persistent memory / resumable runs.** Checkpoint the transcript and run state to disk.
5. **Input/output guardrails** (e.g. prompt-injection screening of file contents before they enter
   the context). This matters as soon as tools read untrusted data.
6. **Process-level sandboxing** (container, seccomp) once tools run code or shell. The path jail is
   enough for fs-only tools, not for `bash`.

**Can wait:** parallel tool execution, streaming tokens to the UI, multi-agent.

## Why not the OpenWorker SDK?
This repo does what the SDK would do for you, so each piece is visible. Once you've built it
by hand, compare it against OpenWorker's harness builder. The concepts map one-to-one: loop,
tools, sandbox, approvals, compaction.
