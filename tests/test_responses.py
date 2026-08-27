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
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
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
