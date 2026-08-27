"""Translation from Chat Completions to the OpenAI Responses API.

The Responses API (used by GitHub Copilot CLI and other modern agents) speaks a
typed, event-driven SSE protocol that is different from Chat Completions'
simple `data: {...}` delta stream. This module translates between the two so
the router can serve Responses-API clients while still talking Chat Completions
to upstream providers.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncGenerator

import httpx


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _build_response_object(
    response_id: str,
    model: str,
    created_at: int,
    output_items: list,
    usage: dict[str, int],
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": model,
        "output": output_items,
        "usage": usage,
    }


def _build_message_item(
    message_id: str,
    role: str = "assistant",
    status: str = "in_progress",
    content: list | None = None,
) -> dict[str, Any]:
    if content is None:
        content = []
    return {
        "id": message_id,
        "type": "message",
        "status": status,
        "role": role,
        "content": content,
    }


def _build_output_text_part(text: str = "") -> dict[str, Any]:
    return {
        "type": "output_text",
        "text": text,
        "annotations": [],
    }


def input_items_to_messages(input_items: list) -> list[dict[str, Any]]:
    """Translate Responses API `input` items into Chat Completions `messages`.

    Responses `input` items look like:
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "..."}]}
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "..."}]}
        {"type": "function_call", ...}
        {"type": "function_call_output", ...}

    We flatten each message item's content parts into a single string (or a
    list of parts if the content is multimodal). Tool-call items are passed
    through as best-effort assistant/tool messages so the upstream model can
    see the tool history.
    """
    messages: list[dict[str, Any]] = []
    for item in input_items:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            role = item.get("role", "user")
            content = item.get("content", "")
            if isinstance(content, list):
                texts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") in (
                        "input_text",
                        "output_text",
                        "text",
                    ):
                        texts.append(part.get("text", ""))
                content = "\n".join(texts)
            messages.append({"role": role, "content": content})
        elif itype == "function_call":
            # Represent a tool call as an assistant message with a tool_calls block.
            name = item.get("name", "")
            arguments = item.get("arguments", "")
            call_id = item.get("call_id", item.get("id", ""))
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ],
            })
        elif itype == "function_call_output":
            call_id = item.get("call_id", "")
            output = item.get("output", "")
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": output,
            })
    return messages


async def translate_stream(
    upstream_response: httpx.Response,
    model: str,
) -> AsyncGenerator[str, None]:
    """Translate a Chat Completions SSE stream into a Responses API SSE stream.

    Yields SSE strings (including ``event: ...\\ndata: ...\\n\\n``) ready to be
    sent to the client.
    """
    response_id = _generate_id("resp")
    message_id = _generate_id("msg")

    created_at = int(time.time())
    sequence_number = 0

    full_text = ""
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def emit_event(event_type: str, data: dict[str, Any]) -> str:
        nonlocal sequence_number
        sequence_number += 1
        data = {**data, "type": event_type, "sequence_number": sequence_number}
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    response_in_progress = _build_response_object(
        response_id=response_id,
        model=model,
        created_at=created_at,
        output_items=[],
        usage=usage,
        status="in_progress",
    )

    yield emit_event("response.created", {"response": response_in_progress})
    yield emit_event("response.in_progress", {"response": response_in_progress})

    message_item = _build_message_item(message_id, status="in_progress", content=[])
    yield emit_event("response.output_item.added", {
        "output_index": 0,
        "item": message_item,
    })

    content_part = _build_output_text_part("")
    yield emit_event("response.content_part.added", {
        "item_id": message_id,
        "output_index": 0,
        "content_index": 0,
        "part": content_part,
    })

    async for line in upstream_response.aiter_lines():
        if not line.strip():
            continue
        if line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue

        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break

        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices", [])
        if not choices:
            continue

        choice = choices[0]
        delta = choice.get("delta", {})
        content_piece = delta.get("content")
        if content_piece:
            full_text += content_piece
            yield emit_event("response.output_text.delta", {
                "item_id": message_id,
                "output_index": 0,
                "content_index": 0,
                "delta": content_piece,
            })

        if "usage" in chunk:
            usage = chunk["usage"]

    final_content_part = _build_output_text_part(full_text)
    final_message_item = _build_message_item(
        message_id,
        status="completed",
        content=[final_content_part],
    )

    yield emit_event("response.output_text.done", {
        "item_id": message_id,
        "output_index": 0,
        "content_index": 0,
        "text": full_text,
    })

    yield emit_event("response.content_part.done", {
        "item_id": message_id,
        "output_index": 0,
        "content_index": 0,
        "part": final_content_part,
    })

    yield emit_event("response.output_item.done", {
        "output_index": 0,
        "item": final_message_item,
    })

    final_response = _build_response_object(
        response_id=response_id,
        model=model,
        created_at=created_at,
        output_items=[final_message_item],
        usage=usage,
        status="completed",
    )

    yield emit_event("response.completed", {"response": final_response})


def translate_non_stream(
    chat_completion: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """Translate a non-streaming Chat Completions JSON into a Response object."""
    response_id = _generate_id("resp")
    message_id = _generate_id("msg")
    created_at = int(time.time())

    choices = chat_completion.get("choices", [])
    if not choices:
        content = ""
    else:
        message = choices[0].get("message", {})
        content = message.get("content", "") or ""

    usage = chat_completion.get("usage", {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    })

    content_part = _build_output_text_part(content)
    message_item = _build_message_item(
        message_id,
        status="completed",
        content=[content_part],
    )

    return _build_response_object(
        response_id=response_id,
        model=model,
        created_at=created_at,
        output_items=[message_item],
        usage=usage,
        status="completed",
    )
