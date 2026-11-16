import pytest

from harness import Agent, AgentConfig, EventBus
from harness.guardrails import ACT_DONT_ASK_MESSAGE, GuardrailConfig, Guardrails, asks_instead_of_acting
from harness.providers.base import STOP_END, STOP_MAX_TOKENS, ModelTurn, ToolCall
from harness.providers.scripted import Script, ScriptedProvider
from harness.sandbox import Sandbox
from harness.tools import ToolRegistry, filesystem_tools


@pytest.fixture
def rails(tmp_path):
    return Guardrails(ToolRegistry(filesystem_tools(Sandbox(tmp_path))), GuardrailConfig(max_result_chars=20))


@pytest.mark.parametrize("text", [
    "I'll create four folders. Shall I go ahead?",
    "Here's my plan: finance, travel, work. Would you like me to proceed with these moves?",
    "Plan ready. Please confirm and I'll start moving files.",
    "Let me know if you want me to continue.",
])
def test_detects_asking_instead_of_acting(text):
    assert asks_instead_of_acting(text)


@pytest.mark.parametrize("text", [
    "Done. Final layout: finance/ (3), travel/ (2), work/ (3), recipes/ (2).",
    "Organised 10 files into 4 folders.",
    "",
])
def test_ignores_normal_final_answers(text):
    assert not asks_instead_of_acting(text)


def test_check_call_trips(rails):
    tool, trip = rails.check_call(ToolCall("1", "rm_rf", {}))
    assert tool is None and "Unknown tool" in trip.message
    _, trip = rails.check_call(ToolCall("2", "read_file", {"nope": 1}))
    assert trip and trip.message.startswith("Invalid arguments")
    _, trip = rails.check_call(ToolCall("3", "read_file", "not json"))
    assert trip and "JSON object" in trip.message
    _, trip = rails.check_call(ToolCall("4", "list_files", {}))
    assert trip is None


def test_repeat_guard_resets_per_run(rails):
    call = ToolCall("1", "list_files", {"path": "."})
    trips = [rails.check_call(call)[1] for _ in range(4)]
    assert [t is None for t in trips] == [True, True, True, False]
    rails.reset()
    assert rails.check_call(call)[1] is None


def test_cap_result(rails):
    assert rails.cap_result("x" * 50).startswith("x" * 20)
    assert "truncated 30 chars" in rails.cap_result("x" * 50)


def test_check_reply(rails):
    assert rails.check_reply(ModelTurn(stop=STOP_MAX_TOKENS, text="partial")).message.startswith("Your previous reply")
    assert rails.check_reply(ModelTurn(stop=STOP_END, text="Done.")) is None
    assert rails.check_reply(ModelTurn(stop=STOP_END, text="Shall I proceed?")).message == ACT_DONT_ASK_MESSAGE
    # Bounded: only one nudge per run, so a false positive costs at most one extra turn.
    assert rails.check_reply(ModelTurn(stop=STOP_END, text="Shall I proceed?")) is None
    rails.reset()
    assert rails.check_reply(ModelTurn(stop=STOP_END, text="Shall I proceed?")) is not None


def _run(tmp_path, turns, **guard):
    provider = ScriptedProvider(turns)
    events = []
    bus = EventBus()
    bus.subscribe(events.append)
    agent = Agent(provider, "sys", ToolRegistry(filesystem_tools(Sandbox(tmp_path))), bus=bus,
                  config=AgentConfig(guardrails=GuardrailConfig(**guard)))
    return agent.run("organise"), provider, events


def test_loop_nudges_a_model_that_asks_then_it_acts(tmp_path):
    turns = [
        Script("I plan to create a 'notes' folder. Shall I go ahead?"),
        Script("Creating it.", [("create_directory", {"path": "notes"})]),
        Script("Done: created notes/."),
    ]
    result, provider, events = _run(tmp_path, turns)
    assert result.status == "completed" and result.final_text == "Done: created notes/."
    assert (tmp_path / "notes").is_dir()
    assert ("user", ACT_DONT_ASK_MESSAGE) in provider.session.transcript
    assert any(e.kind == "guardrail" and "asked a question" in e.data["reason"] for e in events)


def test_nudge_can_be_disabled(tmp_path):
    result, _, _ = _run(tmp_path, [Script("Shall I go ahead?")], nudge_text_questions=False)
    assert result.status == "completed" and result.final_text == "Shall I go ahead?"
