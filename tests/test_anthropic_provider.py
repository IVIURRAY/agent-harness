"""Checks the request shape and append-only transcript without calling the API."""

from types import SimpleNamespace as NS

from harness.providers.anthropic_provider import AnthropicProvider
from harness.providers.base import ToolOutcome
from harness.sandbox import Sandbox
from harness.tools import filesystem_tools


class FakeMessages:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def usage():
    return NS(input_tokens=100, output_tokens=20, cache_read_input_tokens=50, cache_creation_input_tokens=0)


def test_request_shape_and_append_only(tmp_path):
    thinking = NS(type="thinking", thinking="plan", signature="sig")
    tool_use = NS(type="tool_use", id="tu_1", name="list_files", input={"path": "."})
    r1 = NS(content=[thinking, tool_use], stop_reason="tool_use", model="claude-sonnet-5-5", usage=usage())
    r2 = NS(content=[NS(type="text", text="done")], stop_reason="end_turn", model="claude-sonnet-5-5", usage=usage())
    fake = FakeMessages([r1, r2])
    client = NS(beta=NS(messages=fake))

    provider = AnthropicProvider(client=client)
    session = provider.start("sys", filesystem_tools(Sandbox(tmp_path)))
    session.add_user("go")
    turn = session.step()
    assert turn.tool_calls[0].name == "list_files" and turn.thinking == "plan"
    assert turn.usage.context_tokens == 170

    session.add_tool_results([ToolOutcome("tu_1", "(empty)")])
    session.step()

    req = fake.calls[1]
    assert req["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert req["fallbacks"] == "default" and req["betas"] == ["server-side-fallback-2026-07-01"]
    assert all(t["strict"] and t["input_schema"]["additionalProperties"] is False for t in req["tools"])
    assert "tool_choice" not in req
    # Assistant content replayed unchanged - thinking block (with signature) included.
    assert req["messages"][1]["content"] is r1.content
    # First request's messages are a strict prefix of the second's.
    assert fake.calls[1]["messages"][: len(fake.calls[0]["messages"])] == fake.calls[0]["messages"]
    # Tool list is identical across requests (cache + thinking-binding prefix).
    assert fake.calls[0]["tools"] == fake.calls[1]["tools"]
