"""OpenRouter / OpenAI-compatible provider, tested with real SDK response types and no network."""

from types import SimpleNamespace as NS

import pytest
from openai.types.chat import ChatCompletion

from harness import Agent, AgentConfig
from harness.config import load_settings
from harness.factory import build_provider
from harness.providers import openrouter_models
from harness.providers.base import STOP_END, STOP_MAX_TOKENS, STOP_TOOL_USE, ToolOutcome
from harness.providers.openai_compat import OpenAICompatProvider
from harness.sandbox import Sandbox
from harness.tools import ToolRegistry, filesystem_tools


def completion(message, finish="stop", model="m:free", usage=True, **extra):
    data = {
        "id": "gen-1", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", **message}}],
        **extra,
    }
    if usage:
        data["usage"] = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150,
                         "prompt_tokens_details": {"cached_tokens": 20}}
    return ChatCompletion.model_validate(data)


def tool_call(id_, name, args):
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": args}}


class FakeCompletions:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def make(tmp_path, responses, **kw):
    fake = FakeCompletions(responses)
    provider = OpenAICompatProvider(model="m:free", api_key="k", client=NS(chat=NS(completions=fake)), **kw)
    return provider, fake


def test_tool_round_trip_and_request_shape(tmp_path):
    r1 = completion({"content": None, "tool_calls": [tool_call("c1", "list_files", '{"path": "."}')]},
                    finish="tool_calls")
    r2 = completion({"content": "all done"})
    provider, fake = make(tmp_path, [r1, r2], fallback_models=["b:free"])
    session = provider.start("sys", filesystem_tools(Sandbox(tmp_path)))
    session.add_user("go")

    turn = session.step()
    assert turn.stop == STOP_TOOL_USE and turn.tool_calls[0].args == {"path": "."}
    assert turn.usage.input_tokens == 100 and turn.usage.cache_read_tokens == 20

    session.add_tool_results([ToolOutcome("c1", "nope", is_error=True)])
    turn2 = session.step()
    assert turn2.stop == STOP_END and turn2.text == "all done"

    req = fake.calls[1]
    assert req["messages"][0] == {"role": "system", "content": "sys"}
    assert req["messages"][2]["tool_calls"][0]["id"] == "c1"
    assert req["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": "ERROR: nope"}
    assert req["extra_body"] == {"provider": {"require_parameters": True}, "models": ["m:free", "b:free"]}
    assert req["tools"][0]["type"] == "function"
    assert fake.calls[1]["messages"][: len(fake.calls[0]["messages"])] == fake.calls[0]["messages"]


def test_tool_calls_with_finish_reason_stop_still_run(tmp_path):
    r = completion({"content": "", "tool_calls": [tool_call("c1", "list_files", "{}")]}, finish="stop")
    provider, _ = make(tmp_path, [r])
    session = provider.start("sys", [])
    session.add_user("go")
    assert session.step().stop == STOP_TOOL_USE


def test_truncated_tool_calls_are_dropped(tmp_path):
    r = completion({"content": "", "tool_calls": [tool_call("c1", "move_file", '{"source": "a')]}, finish="length")
    provider, _ = make(tmp_path, [r])
    session = provider.start("sys", [])
    session.add_user("go")
    turn = session.step()
    assert turn.stop == STOP_MAX_TOKENS and turn.tool_calls == []


def test_malformed_arguments_become_an_error_the_model_sees(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    r1 = completion({"content": "", "tool_calls": [tool_call("c1", "read_file", "{path: a.txt")]}, finish="tool_calls")
    r2 = completion({"content": "ok"})
    provider, fake = make(tmp_path, [r1, r2])
    agent = Agent(provider, "sys", ToolRegistry(filesystem_tools(Sandbox(tmp_path))), config=AgentConfig())
    assert agent.run("go").status == "completed"
    tool_msg = fake.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["content"].startswith("ERROR: Invalid arguments")


def test_error_body_with_no_choices_is_a_provider_error(tmp_path):
    bad = ChatCompletion.model_construct(id="x", choices=[], model="m", object="chat.completion", created=0)
    provider, _ = make(tmp_path, [bad])
    agent = Agent(provider, "sys", ToolRegistry([]))
    result = agent.run("go")
    assert result.status == "error" and "no choices" in result.error


CATALOG = [
    {"id": "big:free", "context_length": 131072, "pricing": {"prompt": "0", "completion": "0"},
     "supported_parameters": ["tools", "tool_choice"]},
    {"id": "small:free", "context_length": 32768, "pricing": {"prompt": "0", "completion": "0"},
     "supported_parameters": ["tools"]},
    {"id": "notools:free", "context_length": 999999, "pricing": {"prompt": "0", "completion": "0"},
     "supported_parameters": ["temperature"]},
    {"id": "paid/model", "context_length": 200000, "pricing": {"prompt": "0.000001", "completion": "0.000002"},
     "supported_parameters": ["tools"]},
]


def test_free_tool_model_filter():
    assert [m["id"] for m in openrouter_models.free_tool_models(CATALOG)] == ["big:free", "small:free"]


def test_factory_auto_selects_free_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("HARNESS_MODEL", raising=False)
    monkeypatch.setattr(openrouter_models, "fetch_models", lambda: CATALOG)
    built = build_provider(load_settings())
    assert built.provider.model == "big:free"
    assert built.provider.fallback_models == ["small:free"]
    assert built.compact_at_tokens == int(131072 * 0.6)
