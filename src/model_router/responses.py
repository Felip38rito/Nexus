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


def _build_function_call_item(
    item_id: str,
    call_id: str,
    name: str,
    arguments: str,
    status: str = "in_progress",
) -> dict[str, Any]:
    """Build a function_call output item."""
    return {
        "id": item_id,
        "type": "function_call",
        "status": status,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def _normalize_usage(usage: Any) -> dict[str, int]:
    """Normalize upstream usage into the Responses API shape.

    Chat Completions uses prompt_tokens/completion_tokens/total_tokens; the
    Responses API uses input_tokens/output_tokens/total_tokens. The Codex CLI
    strictly requires input_tokens to be present in response.completed.
    """
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {
        "input_tokens": int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
        "output_tokens": int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def translate_tools(tools: list) -> list[dict[str, Any]]:
    """Translate Responses API `tools` (flat) into Chat Completions `tools` (nested).

    Responses: {"name": ..., "description": ..., "parameters": {...}}
    Chat:      {"type": "function", "function": {"name": ..., ...}}
    Accepts both forms defensively (Copilot omits the "type" wrapper).
    """
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            # Already nested (Chat Completions form) — pass through.
            out.append(tool)
            continue
        fn = {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters") or {},
        }
        out.append({"type": "function", "function": fn})
    return out


def _extract_text_from_parts(content: Any) -> str:
    """Extract plain text from a Responses content-part list (or pass a string through)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in (
                "input_text",
                "output_text",
                "text",
            ):
                texts.append(part.get("text", ""))
        return "\n".join(texts)
    return ""


def input_items_to_messages(input_items: list) -> list[dict[str, Any]]:
    """Translate Responses API `input` items into Chat Completions `messages`.

    Coalescing rule (per the reference conversion): consecutive `function_call`
    items fold into a SINGLE assistant message with multiple `tool_calls`,
    and the following `function_call_output` items become `tool` messages that
    must immediately follow it (the Chat API requires each tool_call id to have
    a matching tool message right after). An assistant text followed by calls
    in the same turn merges into one assistant message carrying both content
    and tool_calls.
    """
    messages: list[dict[str, Any]] = []
    pending_assistant: dict[str, Any] | None = None

    def flush_assistant() -> None:
        nonlocal pending_assistant
        if pending_assistant is not None:
            messages.append(pending_assistant)
            pending_assistant = None

    for item in input_items:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")

        if itype == "message":
            role = item.get("role", "user")
            content = _extract_text_from_parts(item.get("content", ""))
            if role == "assistant":
                # Assistant text starts a pending assistant turn (a following
                # function_call may merge into it).
                if pending_assistant is None:
                    pending_assistant = {"role": "assistant", "content": content or None}
                else:
                    # Merge text into the pending assistant message.
                    existing = pending_assistant.get("content") or ""
                    merged = (existing + "\n" + content).strip() if content else existing
                    pending_assistant["content"] = merged or None
            else:
                flush_assistant()
                messages.append({"role": role, "content": content})

        elif itype == "function_call":
            name = item.get("name", "")
            arguments = item.get("arguments", "")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            call_id = item.get("call_id", item.get("id", _generate_id("call")))
            tool_call = {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
            if pending_assistant is None:
                pending_assistant = {"role": "assistant", "content": None, "tool_calls": []}
            pending_assistant.setdefault("tool_calls", []).append(tool_call)

        elif itype == "function_call_output":
            # Tool results must follow their assistant message; flush any open one.
            flush_assistant()
            call_id = item.get("call_id", "")
            output = item.get("output", "")
            if isinstance(output, list):
                output = _extract_text_from_parts(output)
            if not isinstance(output, str):
                output = str(output)
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": output,
            })

        elif itype == "reasoning":
            # Encrypted reasoning content can't be replayed upstream; drop it.
            flush_assistant()
            continue

    flush_assistant()
    return messages


async def translate_stream(
    upstream_response: httpx.Response,
    model: str,
) -> AsyncGenerator[str, None]:
    """Translate a Chat Completions SSE stream into a Responses API SSE stream.

    Implements the state machine described in the ultra plan: it tracks open
    items (message or function_call) and closes them implicitly when a new
    item starts, on `finish_reason`, or on EOF. Tool-call arguments may arrive
    fragmented across many chunks (indexed by `index`); Ollama tends to send
    them in a single chunk, but we tolerate interleaving defensively.

    Yields SSE strings (``event: ...\\ndata: ...\\n\\n``) ready for the client.
    """
    response_id = _generate_id("resp")
    created_at = int(time.time())
    sequence_number = 0

    # Accumulated final state.
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    output_items: list[dict[str, Any]] = []  # completed items, in order

    # Per-item live state.
    # current is either {"kind": "message", ...} or {"kind": "fc", "tool_index": N, ...}
    current: dict[str, Any] | None = None
    next_output_index = 0

    def emit_event(event_type: str, data: dict[str, Any]) -> str:
        nonlocal sequence_number
        sequence_number += 1
        data = {**data, "type": event_type, "sequence_number": sequence_number}
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    def open_message() -> str:
        nonlocal current, next_output_index
        msg_id = _generate_id("msg")
        current = {
            "kind": "message",
            "item_id": msg_id,
            "output_index": next_output_index,
            "text": "",
        }
        item = _build_message_item(msg_id, status="in_progress", content=[])
        events = [
            emit_event("response.output_item.added", {
                "output_index": next_output_index, "item": item,
            }),
            emit_event("response.content_part.added", {
                "item_id": msg_id, "output_index": next_output_index,
                "content_index": 0, "part": _build_output_text_part(""),
            }),
        ]
        next_output_index += 1
        return "".join(events)

    def open_fc(call_id: str, name: str, tool_index: int) -> str:
        nonlocal current, next_output_index
        fc_id = _generate_id("fc")
        current = {
            "kind": "fc",
            "item_id": fc_id,
            "output_index": next_output_index,
            "tool_index": tool_index,
            "call_id": call_id,
            "name": name,
            "args": "",
        }
        item = _build_function_call_item(fc_id, call_id, name, "", status="in_progress")
        events = emit_event("response.output_item.added", {
            "output_index": next_output_index, "item": item,
        })
        next_output_index += 1
        return events

    def close_current() -> str:
        nonlocal current
        if current is None:
            return ""
        events = ""
        if current["kind"] == "message":
            text = current["text"]
            part = _build_output_text_part(text)
            events += emit_event("response.output_text.done", {
                "item_id": current["item_id"], "output_index": current["output_index"],
                "content_index": 0, "text": text,
            })
            events += emit_event("response.content_part.done", {
                "item_id": current["item_id"], "output_index": current["output_index"],
                "content_index": 0, "part": part,
            })
            final_item = _build_message_item(
                current["item_id"], status="completed", content=[part],
            )
            events += emit_event("response.output_item.done", {
                "output_index": current["output_index"], "item": final_item,
            })
            output_items.append(final_item)
        else:  # function_call
            args = current["args"]
            events += emit_event("response.function_call_arguments.done", {
                "item_id": current["item_id"], "output_index": current["output_index"],
                "arguments": args,
            })
            final_item = _build_function_call_item(
                current["item_id"], current["call_id"], current["name"], args,
                status="completed",
            )
            events += emit_event("response.output_item.done", {
                "output_index": current["output_index"], "item": final_item,
            })
            output_items.append(final_item)
        current = None
        return events

    # ---- initial events -----------------------------------------------------
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

    finish_reason: str | None = None

    async for line in upstream_response.aiter_lines():
        if not line.strip() or line.startswith(":") or not line.startswith("data:"):
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
            if "usage" in chunk:
                usage = chunk["usage"]
            continue

        choice = choices[0]
        delta = choice.get("delta", {}) or {}

        # ---- tool calls ----
        tool_calls = delta.get("tool_calls") or []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            idx = tc.get("index", 0)
            call_id = tc.get("id") or ""
            function = tc.get("function", {}) or {}
            name = function.get("name", "")
            arg_piece = function.get("arguments", "")

            # New tool index while another item is open → close + open.
            if current is None:
                yield open_fc(call_id, name, idx)
            elif current["kind"] != "fc" or current["tool_index"] != idx:
                yield close_current()
                yield open_fc(call_id, name, idx)

            # Backfill identity fields if they arrived with this chunk.
            if call_id and not current["call_id"]:
                current["call_id"] = call_id
            if name and not current["name"]:
                current["name"] = name

            if arg_piece:
                current["args"] += arg_piece
                yield emit_event("response.function_call_arguments.delta", {
                    "item_id": current["item_id"],
                    "output_index": current["output_index"],
                    "delta": arg_piece,
                })

        # ---- text content ----
        content_piece = delta.get("content")
        if content_piece:
            if current is None:
                yield open_message()
            elif current["kind"] != "message":
                yield close_current()
                yield open_message()
            current["text"] += content_piece
            yield emit_event("response.output_text.delta", {
                "item_id": current["item_id"],
                "output_index": current["output_index"],
                "content_index": 0,
                "delta": content_piece,
            })

        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

        if "usage" in chunk:
            usage = chunk["usage"]

    # ---- finalize ----
    yield close_current()

    status = "completed" if finish_reason != "length" else "incomplete"
    final_response = _build_response_object(
        response_id=response_id,
        model=model,
        created_at=created_at,
        output_items=output_items,
        usage=_normalize_usage(usage),
        status=status,
    )
    yield emit_event("response.completed", {"response": final_response})


def translate_non_stream(
    chat_completion: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """Translate a non-streaming Chat Completions JSON into a Response object."""
    response_id = _generate_id("resp")
    created_at = int(time.time())

    choices = chat_completion.get("choices", [])
    output_items: list[dict[str, Any]] = []

    if choices:
        message = choices[0].get("message", {}) or {}
        content = message.get("content", "")
        tool_calls = message.get("tool_calls") or []

        # Text message first (only if non-empty).
        if content:
            item_id = _generate_id("msg")
            part = _build_output_text_part(content)
            output_items.append(_build_message_item(
                item_id, status="completed", content=[part],
            ))

        # Then the function_call items.
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            call_id = tc.get("id") or _generate_id("call")
            function = tc.get("function", {}) or {}
            name = function.get("name", "")
            arguments = function.get("arguments", "")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            output_items.append(_build_function_call_item(
                _generate_id("fc"), call_id, name, arguments, status="completed",
            ))

    usage = chat_completion.get("usage", {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    })

    finish = choices[0].get("finish_reason") if choices else None
    status = "completed"
    if finish == "length":
        status = "incomplete"

    return _build_response_object(
        response_id=response_id,
        model=model,
        created_at=created_at,
        output_items=output_items,
        usage=_normalize_usage(usage),
        status=status,
    )
