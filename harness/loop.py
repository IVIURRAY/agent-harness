"""The agent loop.

    task -> [compact?] -> model -> tool calls? -> (approve) -> execute -> results -> model ...
                                     \-> no: done

Everything the LLM can't be trusted to do itself lives here: step limits, input validation,
sandboxed execution, approvals, result truncation, repeated-call detection, compaction, and
emitting an event for every transition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from . import events as ev
from .approvals import ApprovalGate, AutoApprove
from .events import EventBus
from .providers.base import (
    STOP_MAX_TOKENS,
    STOP_PAUSE,
    STOP_REFUSAL,
    ModelTurn,
    Provider,
    ProviderError,
    ToolCall,
    ToolOutcome,
)
from .tools.base import ToolError, ToolRegistry, truncate

SUMMARY_PROMPT = """\
Context is getting long, so the conversation will be replaced by a summary. Write that summary now.
Retain: the original goal; every action already taken and its result (including any the user declined,
and why); the current state of the workspace as you last observed it; decisions made and the plan for
the remaining work. Be concise and factual. Reply with the summary text only - do not call any tools."""

RESUME_TEMPLATE = """\
{task}

<progress_summary>
The conversation so far was compacted. Summary of what has happened:
{summary}
</progress_summary>

Continue the task from where you left off. Re-check the current state with tools before acting."""


@dataclass
class AgentConfig:
    max_steps: int = 40                 # model calls per run - the hard stop on runaway loops
    compact_at_tokens: int = 120_000    # compact when the context reaches this size
    max_identical_calls: int = 3        # same tool + same args this many times in a row -> blocked


@dataclass
class RunResult:
    status: str                         # completed | max_steps | refused | error
    final_text: str = ""
    steps: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    compactions: int = 0
    error: str = ""


@dataclass
class _RunState:
    approve_rest: bool = False
    last_call_sig: str = ""
    repeat_count: int = 0
    context_tokens: int = 0
    result: RunResult = field(default_factory=lambda: RunResult(status="running"))


class Agent:
    def __init__(
        self,
        provider: Provider,
        system: str,
        tools: ToolRegistry,
        approvals: ApprovalGate | None = None,
        bus: EventBus | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.provider = provider
        self.system = system
        self.tools = tools
        self.approvals = approvals or AutoApprove()
        self.bus = bus or EventBus()
        self.config = config or AgentConfig()

    # ----------------------------------------------------------------------------------
    def run(self, task: str) -> RunResult:
        cfg, bus = self.config, self.bus
        state = _RunState()
        res = state.result
        session = self.provider.start(self.system, list(self.tools))
        session.add_user(task)
        bus.emit(ev.RUN_STARTED, task=task, provider=self.provider.name, model=self.provider.model,
                 tools=[t.name for t in self.tools])

        for step in range(1, cfg.max_steps + 1):
            res.steps = step

            if state.context_tokens >= cfg.compact_at_tokens:
                self._compact(session, task, state)

            bus.emit(ev.STATUS, step=step, max_steps=cfg.max_steps, phase="thinking",
                     detail=f"waiting for the model (context ≈ {state.context_tokens:,} tokens)")
            bus.emit(ev.MODEL_REQUEST, step=step)
            try:
                turn = session.step()
            except ProviderError as exc:
                return self._finish(res, "error", error=str(exc))

            self._record_turn(turn, step, state)

            if turn.stop == STOP_PAUSE:
                continue
            if turn.stop == STOP_REFUSAL:
                return self._finish(res, "refused", final_text=turn.text, error="; ".join(turn.notes))
            if turn.stop == STOP_MAX_TOKENS and not turn.tool_calls:
                bus.emit(ev.GUARDRAIL, reason="reply hit max_tokens; asking the model to continue concisely")
                session.add_user("Your previous reply was cut off by the output limit. Continue, more concisely.")
                continue
            if not turn.tool_calls:
                return self._finish(res, "completed", final_text=turn.text)

            bus.emit(ev.STATUS, step=step, max_steps=cfg.max_steps, phase="acting",
                     detail=f"model requested {len(turn.tool_calls)} tool call(s)")
            results = [self._execute(call, state) for call in turn.tool_calls]
            res.tool_calls += len(results)
            session.add_tool_results(results)

        return self._finish(res, "max_steps", error=f"Stopped after {cfg.max_steps} model calls")

    # ----------------------------------------------------------------------------------
    def _record_turn(self, turn: ModelTurn, step: int, state: _RunState) -> None:
        res = state.result
        res.input_tokens += turn.usage.input_tokens + turn.usage.cache_read_tokens + turn.usage.cache_write_tokens
        res.output_tokens += turn.usage.output_tokens
        state.context_tokens = turn.usage.context_tokens
        self.bus.emit(ev.MODEL_RESPONSE, step=step, stop=turn.stop, model=turn.model,
                      tool_calls=len(turn.tool_calls), notes=turn.notes,
                      usage=vars(turn.usage), context_tokens=state.context_tokens)
        if turn.thinking:
            self.bus.emit(ev.THINKING, step=step, text=turn.thinking)
        if turn.text:
            self.bus.emit(ev.ASSISTANT_TEXT, step=step, text=turn.text)

    def _execute(self, call: ToolCall, state: _RunState) -> ToolOutcome:
        bus = self.bus
        tool = self.tools.get(call.name)
        bus.emit(ev.TOOL_CALL, id=call.id, name=call.name, args=call.args)

        def fail(msg: str) -> ToolOutcome:
            bus.emit(ev.TOOL_RESULT, id=call.id, name=call.name, ok=False, output=msg)
            return ToolOutcome(call.id, msg, is_error=True)

        if tool is None:
            return fail(f"Unknown tool {call.name!r}.")
        if problem := self.tools.validate(tool, call.args):
            return fail(f"Invalid arguments: {problem}")
        args: dict[str, Any] = call.args

        # Repeated-call guard: catches a model stuck retrying the same thing.
        sig = f"{call.name}:{json.dumps(args, sort_keys=True)}"
        state.repeat_count = state.repeat_count + 1 if sig == state.last_call_sig else 1
        state.last_call_sig = sig
        if state.repeat_count > self.config.max_identical_calls:
            bus.emit(ev.GUARDRAIL, reason=f"blocked repeated call {call.name}")
            return fail("Blocked by harness: this exact call was repeated too many times. Try something different.")

        if tool.requires_approval and not state.approve_rest:
            preview = tool.preview(args)
            bus.emit(ev.STATUS, phase="awaiting approval", detail=preview)
            bus.emit(ev.APPROVAL_REQUESTED, id=call.id, name=call.name, preview=preview)
            decision = self.approvals.decide(tool, args, preview)
            state.approve_rest = state.approve_rest or decision.approve_rest
            bus.emit(ev.APPROVAL_DECIDED, id=call.id, approved=decision.approved, reason=decision.reason)
            if not decision.approved:
                why = f" Reason: {decision.reason.rstrip('.')}." if decision.reason else ""
                return fail(f"The user declined this operation.{why} Do not retry it unchanged.")

        bus.emit(ev.STATUS, phase="executing", detail=tool.preview(args))
        try:
            output = truncate(tool.fn(**args))
        except ToolError as exc:
            return fail(f"Error: {exc}")
        except Exception as exc:  # never leak stack traces / internals into the context
            bus.emit(ev.GUARDRAIL, reason=f"tool {call.name} crashed: {exc!r}")
            return fail("Internal tool error.")
        bus.emit(ev.TOOL_RESULT, id=call.id, name=call.name, ok=True, output=output)
        return ToolOutcome(call.id, output)

    def _compact(self, session, task: str, state: _RunState) -> None:
        before = state.context_tokens
        self.bus.emit(ev.STATUS, phase="compacting", detail=f"context ≈ {before:,} tokens - summarising history")
        try:
            summary = session.summarize(SUMMARY_PROMPT)
        except ProviderError as exc:
            self.bus.emit(ev.GUARDRAIL, reason=f"compaction failed, continuing uncompacted: {exc}")
            return
        session.reset_with(RESUME_TEMPLATE.format(task=task, summary=summary))
        state.context_tokens = 0
        state.result.compactions += 1
        self.bus.emit(ev.COMPACTION, before_tokens=before, summary=summary)

    def _finish(self, res: RunResult, status: str, final_text: str = "", error: str = "") -> RunResult:
        res.status, res.final_text, res.error = status, final_text, error
        self.bus.emit(ev.RUN_FINISHED, **vars(res))
        return res
