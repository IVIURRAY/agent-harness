from harness import Agent, AgentConfig, EventBus
from harness.guardrails import GuardrailConfig
from harness.approvals import ScriptedApprovals
from harness.providers.scripted import Script, ScriptedProvider
from harness.sandbox import Sandbox
from harness.tools import ToolRegistry, filesystem_tools


def make_agent(tmp_path, turns, approvals=None, max_identical_calls=3, **cfg):
    (tmp_path / "a.txt").write_text("a")
    provider = ScriptedProvider(turns)
    events = []
    bus = EventBus()
    bus.subscribe(events.append)
    agent = Agent(provider, "sys", ToolRegistry(filesystem_tools(Sandbox(tmp_path))),
                  approvals=approvals or ScriptedApprovals([]), bus=bus,
                  config=AgentConfig(guardrails=GuardrailConfig(max_identical_calls=max_identical_calls), **cfg))
    return agent, provider, events


def test_declined_operation_is_not_executed_and_model_is_told(tmp_path):
    approvals = ScriptedApprovals([False])
    agent, provider, _ = make_agent(tmp_path, [Script(calls=[("create_directory", {"path": "x"})])], approvals)
    result = agent.run("go")
    assert result.status == "completed"
    assert not (tmp_path / "x").exists()
    assert approvals.seen == ["Create directory  x/"]
    outcome = provider.session.results[0]
    assert outcome.is_error and "declined" in outcome.content


def test_read_only_tools_skip_approval(tmp_path):
    approvals = ScriptedApprovals([])
    agent, _, _ = make_agent(tmp_path, [Script(calls=[("list_files", {})])], approvals)
    agent.run("go")
    assert approvals.seen == []


def test_unknown_tool_and_bad_args_become_errors(tmp_path):
    agent, provider, _ = make_agent(tmp_path, [Script(calls=[("rm_rf", {}), ("read_file", {"nope": 1})])])
    agent.run("go")
    assert all(r.is_error for r in provider.session.results)


def test_repeated_identical_calls_are_blocked(tmp_path):
    turns = [Script(calls=[("read_file", {"path": "a.txt"})]) for _ in range(5)]
    agent, provider, _ = make_agent(tmp_path, turns, max_identical_calls=3)
    agent.run("go")
    flags = [r.is_error for r in provider.session.results]
    assert flags == [False, False, False, True, True]


def test_max_steps_stops_runaway_loop(tmp_path):
    turns = [Script(calls=[("list_files", {"path": "."})]) for _ in range(10)]
    agent, _, _ = make_agent(tmp_path, turns, max_steps=3, max_identical_calls=99)
    assert agent.run("go").status == "max_steps"


def test_compaction_replaces_history_with_summary(tmp_path):
    turns = [Script(calls=[("list_files", {})], context_tokens=5_000), Script("done")]
    agent, provider, events = make_agent(tmp_path, turns, compact_at_tokens=4_000)
    result = agent.run("organise")
    assert result.compactions == 1
    first = provider.session.transcript[0]
    assert first[0] == "user" and "organise" in first[1] and "<progress_summary>" in first[1]
    assert any(e.kind == "compaction" for e in events)


def test_decline_reason_is_passed_to_the_model(tmp_path):
    class WithReason:
        def decide(self, tool, args, preview):
            from harness.approvals import Decision
            return Decision(False, "keep receipts separate")

    agent, provider, _ = make_agent(tmp_path, [Script(calls=[("create_directory", {"path": "x"})])], WithReason())
    agent.run("go")
    assert "Reason: keep receipts separate. Do not retry" in provider.session.results[0].content


def test_scripted_turns_can_branch_on_a_decline(tmp_path):
    def declined(history):
        return any(o.is_error and "declined" in o.content for _, o in history)

    turns = [Script(calls=[("create_directory", {"path": "x"})]),
             Script("adapted", when=declined),
             Script("approved path")]
    agent, _, _ = make_agent(tmp_path, turns, ScriptedApprovals([False]))
    assert agent.run("go").final_text == "adapted"
