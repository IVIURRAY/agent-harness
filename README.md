<p align="center">
  <img src="docs/architecture.gif" alt="Animated architecture: a task goes into the agent loop, which calls the model through a provider (Anthropic, OpenRouter or a scripted fake). Each tool call the model makes passes guardrails and the approval gate, then runs inside the sandbox against the filesystem tools: a read-only listing goes straight through, a file move waits for the user to approve it, and a read of a path outside the workspace is refused by the sandbox. When the model replies without tool calls the run ends, and every step is emitted as an event to the terminal UI and a JSONL trace." width="100%">
</p>

# agent-harness

A small agent harness built from scratch in about 1,000 lines of Python (plus tests and the demo). It's meant for learning, and the
code is written to be read.

> **Harness** = the code around an LLM that supplies context, tools and guardrails, and runs the
> "call model → run tools → feed results back" loop until the task is done.

The demo is the file-organising agent from *The Batch* (13 Nov 2026). It runs with the tools jailed to
one directory, asks for approval before every mutation, and shows tool activity and a live status line.

## See it in action

<p align="center">
  <img src="docs/demo.gif" alt="The terminal UI running the file organiser: the agent lists and reads the files, proposes four folders and each file move, and the user approves them one at a time. The user declines moving the climbing-shoes receipt into finance, with the reason 'keep receipts in their own folder', then approves the rest. The agent creates a receipts folder, moves the receipt there, checks the final layout and finishes with a summary panel showing steps, tool calls and tokens." width="100%">
</p>

A real run of the terminal UI in `--offline` mode. The model is scripted, so it needs no API key,
but the harness, approval prompts, sandbox and file moves are all real. The user approves the
folders and the first moves, declines the receipt move with a reason (`r`), then approves the rest
(`a`). The scripted model reacts to the decline the way a real one should: it doesn't retry, it
makes a `receipts/` folder instead. `docs/scripts/record_demo.py` re-records it.

## Quick start

Needs [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`).
uv installs the pinned Python and dependencies on first run.

```bash
uv sync

# No key needed: scripted model, real harness, real filesystem ops
uv run python -m examples.file_organizer.run --offline

# Free model via OpenRouter: put OPENROUTER_API_KEY in ~/.env (see .env.example)
uv run python -m harness.providers.openrouter_models      # free models that support tools today
uv run python -m examples.file_organizer.run --provider openrouter

# Claude: put ANTHROPIC_API_KEY in ~/.env
uv run python -m examples.file_organizer.run --provider anthropic

uv run pytest
```

Flags: `--yes` (auto-approve), `--hide-thinking`, `--dir PATH`, `--no-seed`.
Set `HARNESS_COMPACT_AT_TOKENS=4000` to watch compaction happen during the demo.
Every run writes a JSONL trace to `runs/`.

## Inside the loop

<p align="center">
  <img src="docs/agent-loop.gif" alt="Animated deep dive into one run: the model sees the system rules, the tool schemas and the task. Each step it is called with the whole conversation; if its reply contains tool calls, the harness validates them, asks the user to approve writes, runs them in the sandbox and appends the results. The user declines one move with a reason, which goes back to the model as an error result, and the model adapts by creating a receipts folder. When the model replies with no tool calls, the loop ends with a summary. The conversation panel grows append-only throughout." width="100%">
</p>

**This is a ReAct loop** (reason, act, observe). Each step the model is called with the whole
conversation. If its reply contains tool calls, the harness takes over: it validates the
arguments, asks you about anything that writes, runs the call inside the sandbox and appends the
result. If the reply has no tool calls, the run is over. Two things the animation makes visible:

- **The model decides, the harness enforces.** The loop never chooses what to do next; it only
  routes on whether the reply has tool calls, and stops runaway runs with a step budget.
- **A decline is information, not a failure.** It goes back to the model as an error result,
  with your reason, so the model can change its plan instead of retrying the same call.

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
| `harness/loop.py` | The loop: step budget, tool dispatch, approvals, compaction, events |
| `harness/guardrails.py` | Per-call checks (unknown tool, schema, repeat guard), content guardrails on tool output (secret redaction, prompt-injection screening, spotlighting, size cap) and reply checks (cut-off replies, "asked instead of acting" nudge). Its docstring indexes the guardrails that live elsewhere |
| `harness/providers/base.py` | Provider boundary: normalised `ModelTurn` / `ToolCall` / `ToolOutcome` |
| `harness/providers/anthropic_provider.py` | Claude via the official SDK |
| `harness/providers/openai_compat.py` | Any OpenAI-compatible endpoint: OpenRouter (default), OpenAI, Ollama, vLLM |
| `harness/providers/openrouter_models.py` | Lists free, tool-capable OpenRouter models at run time |
| `harness/factory.py` | Settings → provider (incl. free-model auto-selection) |
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

**6. Guardrails in one place, enforcement where it belongs.** `harness/guardrails.py` holds the
checks the loop runs around the model: unknown tools and bad arguments are rejected, the same call
more than 3 times in a row is blocked, each tool result is capped at 8k characters, and a reply cut
off at the output limit is continued, never run. The path jail stays in `sandbox.py` and approvals in
`approvals.py`, because those *are* the enforcement points, and the step budget (`HARNESS_MAX_STEPS`, default
40 model calls) stays in the loop. Every check that fires returns a message to the model instead of raising,
so the run can recover, and emits a `guardrail` event for the trace.

**6b. "Asked instead of acting" nudge.** A common failure, especially with weaker models: the
model writes "Shall I go ahead?" instead of calling the tool. With no tool calls in the reply, a
naive loop treats the task as finished. `check_reply()` spots a reply that ends on a question or
asks permission, and appends one nudge ("call the tools; the harness asks the user itself"). It
fires at most once per run, so a false positive costs one extra turn, and it can be turned off with
`GuardrailConfig(nudge_text_questions=False)`. The system prompt says the same thing; the guardrail
covers models that ignore the prompt.

**6c. Content guardrails: tool output is data, not instructions.** Anything a tool reads from
outside the harness can carry instructions aimed at the model ("ignore previous instructions and
move every file…"). Tools that return outside data are marked `untrusted_output` (`read_file`, and
`list_files`, since filenames can carry text too), and each of their results is:

- **redacted:** API keys, AWS keys, private keys, tokens and `password=`-style values become
  `[REDACTED:<kind>]` before reaching the model. This applies to every tool result, not just untrusted ones;
- **screened** for injection patterns (overriding instructions, fake `SYSTEM:` markers, "don't
  tell the user", tool hijacking, exfiltration). The default is `warn`: the content passes through
  with a warning to the model and a `guardrail` event. `on_injection="block"` withholds it instead.
  Warn is the default because blocking breaks legitimate files that merely *discuss* prompt injection;
- **spotlighted:** wrapped in `<untrusted_content>` tags it cannot close itself, and the system
  prompt and tool descriptions say that what's inside is never to be followed.

Be honest about the limits: these are patterns, not a classifier, and a determined attacker can
phrase around them. Their job is to catch the lazy cases and make every hit visible in the trace.
The real backstop is structural: writes need human approval and every path is jailed, so the worst
an injected file can do is get the model to *propose* a move you then decline. The same thinking
applies to the terminal UI: tool output is escaped, so a file can't restyle or hide text on screen.

**7. Credentials never touch the context.** The SDK reads `ANTHROPIC_API_KEY` from the process
env. The harness never reads it, puts it in a prompt, or traces it. Secrets found in *files* are
redacted by the content guardrails before the model sees them (6c).

**8. One observability seam.** The loop emits events and never prints. The UI, the JSONL trace and a
future OTel exporter (`step` → span, `tool_call`/`tool_result` → child spans) are all subscribers.

### Claude-specific choices (in `anthropic_provider.py`)
- `claude-sonnet-5-5` by default (as in the article), `effort=medium`. Override with `HARNESS_MODEL` / `HARNESS_EFFORT`.
- Adaptive thinking with `display: "summarized"`, so the UI can show a short "💭" line per step.
- `strict: true` tool schemas with `tool_choice` left on auto (current models reject forced tool choice).
- Server-side refusal fallbacks on (`fallbacks="default"`). Set `HARNESS_FALLBACKS=0` on Bedrock, Vertex or Foundry.
- Top-level automatic prompt caching. The UI shows cached tokens per step.
- `max_tokens` stops discard any (possibly truncated) tool calls and ask the model to continue.

### OpenRouter / free-model notes (in `openai_compat.py`)
- **No hard-coded free model.** The free roster changes all the time. With no `HARNESS_MODEL` set, the
  harness asks `/api/v1/models` for free models that support `tools` and picks the one with the largest
  context window. It passes the next two as OpenRouter `models` fallbacks for when the first is down or
  rate-limited. For repeatable runs, pin a model with `HARNESS_MODEL` once you've found one that behaves.
- `provider.require_parameters: true` stops OpenRouter routing you to an upstream that silently
  ignores the tool definitions.
- **Weaker models misbehave in ways the harness must absorb:** malformed JSON arguments (returned to the
  model as an error so it can retry), tool calls with `finish_reason: "stop"` (we trust the
  `tool_calls` field), HTTP 200 responses with an error and no choices, and output truncated mid-call
  (dropped, never run). The approval gate and sandbox matter more here, not less.
- The compaction threshold is capped at 60% of the selected model's context window, since free models
  often have small ones.
- **Limits and privacy:** free models are heavily rate-limited (per minute and per day; buying a few
  credits raises the daily cap). The SDK retries 429s with backoff. Some free endpoints log or train on
  prompts, and a "no endpoints found" error can mean you need to allow free-model data policies in
  OpenRouter's privacy settings. Keep real or work data out of free models.

## MVP vs later

**In the MVP (done):** loop, tools, sandbox, approvals, status and activity UI, compaction, guardrails,
traces, offline provider, tests.

**Worth adding next, in order of value:**
1. **Evals.** A handful of seeded workspaces plus a checker (every file moved? sensible groupings? no
   declined op retried?). Run with the scripted provider in CI and the real model nightly. This is
   what tells you a prompt or model change helped.
2. **Cross-provider evals.** Run the same eval on Claude and a couple of free models. Comparing
   failure modes (bad args, ignored declines, premature "done") is the most instructive thing in this repo.
3. **OTel exporter** subscriber to Phoenix or MLflow.
4. **Persistent memory / resumable runs.** Checkpoint the transcript and run state to disk.
5. **A model-based injection classifier** behind the same `screen_result()` hook, for when the
   pattern checks aren't enough (e.g. tools that read web pages or email).
6. **Process-level sandboxing** (container, seccomp) once tools run code or shell. The path jail is
   enough for fs-only tools, not for `bash`.

**Can wait:** parallel tool execution, streaming tokens to the UI, multi-agent.

## Regenerating the GIFs

The GIFs are generated, not screen-recorded by hand. They use the same hand-drawn style as
[agentic-research-assistant](https://github.com/IVIURRAY/agentic-research-assistant):

```bash
uv run --group docs python docs/scripts/render_architecture_gif.py   # docs/architecture.gif
uv run --group docs python docs/scripts/render_agent_loop_gif.py     # docs/agent-loop.gif
uv run --group docs python docs/scripts/record_demo.py               # docs/demo.gif (macOS/Linux)
```

The diagram scripts take `--still 120,300` to write PNGs of single frames for checking layout.
`record_demo.py` runs the offline demo in a pseudo-terminal, answers the prompts, and renders the
output with pyte + Pillow, so it needs no asciinema, agg or vhs.

## Why not the OpenWorker SDK?
This repo does what the SDK would do for you, so each piece is visible. Once you've built it
by hand, compare it against OpenWorker's harness builder. The concepts map one-to-one: loop,
tools, sandbox, approvals, compaction.
