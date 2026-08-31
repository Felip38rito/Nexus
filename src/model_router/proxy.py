"""OpenAI-compatible relay: /v1/chat/completions + /v1/responses + /v1/models.

The router accepts a request, picks the cheapest adequate tier,
and streams the completion back from an upstream provider under that tier's model id.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .classify import classify
from .config import Settings
from .models import Tier
from . import responses as responses_translate

log = logging.getLogger("model_router.proxy")

router = APIRouter()


def _model_list_payload(models: "RouterModels") -> dict[str, Any]:
    """Advertise the virtual + tier model ids the router understands."""
    data = [
        {
            "id": "adaptive",
            "object": "model",
            "created": 0,
            "owned_by": "axon",
        }
    ]
    for tier in Tier:
        spec = models.tiers[tier]
        data.append(
            {
                "id": tier.value,
                "object": "model",
                "created": 0,
                "owned_by": "axon",
                "tier": tier.value,
                "model": spec.api_id,
            }
        )
    return {"object": "list", "data": data}


def _auth_ok(settings: Settings, request: Request) -> bool:
    if not settings.require_auth:
        return True
    auth = request.headers.get("authorization", "")
    expected = f"Bearer {settings.require_auth}"
    return hmac.compare_digest(auth, expected)


@router.get("/v1/models")
async def list_models(request: Request):
    settings: Settings = request.app.state.settings
    if not _auth_ok(settings, request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return _model_list_payload(settings.models)


async def _process_chat(
    request: Request,
    body: dict[str, Any],
    settings: Settings,
    responses_mode: bool = False,
) -> StreamingResponse:
    """Core routing and forwarding logic for all chat-like endpoints."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="'messages' must be a non-empty list")

    # Classify ONLY the LAST user message — the user's current intent.
    last_user_content: str | None = None
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            last_user_content = content
        elif isinstance(content, list):
            texts = [
                str(c.get("text", ""))
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            ]
            if texts:
                last_user_content = "\n".join(texts)

    if last_user_content is not None:
        prompt = last_user_content
    else:
        parts: list[str] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
        prompt = "\n".join(parts)

    requested_model = body.get("model", "")
    known_tier = settings.models.tier_for_api_id(requested_model)
    if known_tier is None:
        try:
            known_tier = Tier(requested_model)
        except ValueError:
            known_tier = None

    if known_tier is not None:
        routed_tier = known_tier
    else:
        routed_tier = await classify(prompt, settings)

    routed_spec = settings.models.tiers[routed_tier]
    routed_model = routed_spec.api_id
    provider = settings.models.provider_for(routed_spec.provider)

    snippet = prompt[:50].replace("\n", " ") + "..."
    log.info(
        "Tier=%s Model=%s Provider=%s Prompt=%s",
        routed_tier.value,
        routed_model,
        provider.base_url,
        snippet,
    )

    upstream_body = dict(body)
    upstream_body["model"] = routed_model

    target_url = f"{provider.base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ.get(provider.api_key_env, '')}",
        "Content-Type": "application/json",
    }

    stream = bool(body.get("stream", False))
    upstream_body["stream"] = stream

    upstream = getattr(request.app.state, "http_client", None)
    own_client = upstream is None
    if own_client:
        upstream = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=15.0))

    try:
        response = await upstream.post(target_url, headers=headers, json=upstream_body)
    except httpx.HTTPError as exc:
        if own_client:
            await upstream.aclose()
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")

    if response.status_code >= 400:
        detail = response.text[:2000]
        if own_client:
            await upstream.aclose()
        raise HTTPException(status_code=response.status_code, detail=detail)

    async def passthrough():
        try:
            if responses_mode:
                if stream:
                    async for sse in responses_translate.translate_stream(response, routed_model):
                        yield sse.encode("utf-8")
                else:
                    data = response.json()
                    obj = responses_translate.translate_non_stream(data, routed_model)
                    yield json.dumps(obj).encode("utf-8")
            else:
                if stream:
                    async for chunk in response.aiter_bytes():
                        yield chunk
                else:
                    yield response.content
        finally:
            if own_client:
                await upstream.aclose()

    headers_out = {
        "X-Router-Model": routed_model,
        "X-Router-Tier": routed_tier.value,
    }

    media = "text/event-stream" if stream else "application/json"
    return StreamingResponse(
        passthrough(),
        media_type=media,
        headers=headers_out,
    )


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    settings: Settings = request.app.state.settings
    if not _auth_ok(settings, request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    return await _process_chat(request, body, settings)


@router.post("/v1/responses")
async def responses_shim(request: Request):
    settings: Settings = request.app.state.settings
    if not _auth_ok(settings, request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Translation: /v1/responses -> /v1/chat/completions
    # Copilot sends 'instructions' (system) and 'input' (list of items).
    # Older/other clients may send 'messages' directly.
    instructions = body.get("instructions", "")
    input_items = body.get("input")
    messages = body.get("messages")

    chat_messages = []
    if instructions:
        chat_messages.append({"role": "system", "content": instructions})

    if isinstance(input_items, list):
        chat_messages.extend(responses_translate.input_items_to_messages(input_items))
    elif isinstance(messages, list):
        chat_messages.extend(messages)

    chat_body = dict(body)
    chat_body["messages"] = chat_messages

    # Translate tools flat -> nested and attach them to the Chat body.
    tools = body.get("tools")
    if isinstance(tools, list):
        chat_body["tools"] = responses_translate.translate_tools(tools)
    # parallel_tool_calls pass-through
    if "parallel_tool_calls" in body:
        chat_body["parallel_tool_calls"] = body["parallel_tool_calls"]

    # Ask upstream to include usage in the final chunk (needed for metering).
    if chat_body.get("stream"):
        chat_body["stream_options"] = {"include_usage": True}

    return await _process_chat(request, chat_body, settings, responses_mode=True)
