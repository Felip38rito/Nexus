"""Tests for the Chat Completions -> Responses API translation."""
import json

import httpx
import pytest

from model_router import responses as rt


def _sse_chunks() -> list[str]:
    return [
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        "data: [DONE]\n\n",
    ]


class _FakeResponse:
    def __init__(self, lines: list[str]):
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


@pytest.mark.asyncio
async def test_translate_stream_emits_typed_events():
    events = []
    async for sse in rt.translate_stream(_FakeResponse(_sse_chunks()), "test-model"):
        events.append(sse)

    # Parse each SSE frame into (event_type, data)
    parsed = []
    for frame in events:
        lines = frame.strip().split("\n")
        event_type = lines[0].split(":", 1)[1].strip()
        data = json.loads(lines[1].split(":", 1)[1].strip())
        parsed.append((event_type, data))

    types = [t for t, _ in parsed]
    assert types == [
        "response.created",
        "response.in_progress",
        # message item opens lazily when the first text delta arrives
        "response.output_item.added",
        "response.output_text.delta",
        "response.output_text.delta",
        # on finish_reason, the message item closes
        "response.output_text.done",
        "response.completed",
    ]

    # The completed event carries the full text and usage.
    completed = parsed[-1][1]
    assert completed["response"]["status"] == "completed"
    assert completed["response"]["model"] == "test-model"
    output = completed["response"]["output"]
    assert output[0]["type"] == "message"
    assert output[0]["content"][0]["text"] == "Hello world"


def test_translate_non_stream():
    chat = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi there"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    obj = rt.translate_non_stream(chat, "test-model")
    assert obj["object"] == "response"
    assert obj["status"] == "completed"
    assert obj["model"] == "test-model"
    assert obj["output"][0]["content"][0]["text"] == "Hi there"
    assert obj["usage"]["total_tokens"] == 7


def test_translate_non_stream_with_tool_calls():
    chat = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    obj = rt.translate_non_stream(chat, "test-model")
    assert obj["status"] == "completed"
    assert len(obj["output"]) == 1
    item = obj["output"][0]
    assert item["type"] == "function_call"
    assert item["name"] == "bash"
    assert item["arguments"] == "{\"command\":\"ls\"}"
    assert item["call_id"] == "call_abc"


def test_input_items_to_messages_coalescing():
    items = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Do X"}]},
        {"type": "function_call", "call_id": "call_1", "name": "bash", "arguments": "{\"command\":\"ls\"}"},
        {"type": "function_call", "call_id": "call_2", "name": "view", "arguments": "{\"path\":\"/a\"}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "file.txt"},
        {"type": "function_call_output", "call_id": "call_2", "output": "content of /a"},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Now do Y"}]},
    ]
    msgs = rt.input_items_to_messages(items)
    # user, assistant(with 2 tool_calls), tool1, tool2, user
    assert len(msgs) == 5
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert len(msgs[1]["tool_calls"]) == 2
    assert msgs[2]["role"] == "tool" and msgs[2]["tool_call_id"] == "call_1"
    assert msgs[3]["role"] == "tool" and msgs[3]["tool_call_id"] == "call_2"
    assert msgs[4]["role"] == "user"


def test_translate_tools_flat_to_nested():
    tools = [
        {"name": "bash", "description": "runs bash", "parameters": {"type": "object"}},
        {"type": "function", "function": {"name": "already-nested", "parameters": {}}},
    ]
    out = rt.translate_tools(tools)
    assert out[0] == {"type": "function", "function": {"name": "bash", "description": "runs bash", "parameters": {"type": "object"}}}
    assert out[1] == tools[1]  # pass-through

