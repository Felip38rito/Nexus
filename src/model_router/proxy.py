"""OpenAI-compatible relay: /v1/chat/completions + /v1/models.

The router accepts an OpenAI-format request, picks the cheapest adequate tier,
and streams the completion back from Ollama Cloud under that tier's model id.
"""
from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .classify import classify
from .config import Settings

router = APIRouter()


def _model_list_payload(models: "RouterModels") -> dict[str, Any]:
    data = [
        {
            "id": spec.api_id,
            "object": "model",
            "created": 0,
            "owned_by": "ollama-cloud",
        }
        for spec in models.tiers.values()
    ]
    return {"object": "list", "data": data}


def _auth_ok(settings: Settings, request: Request) -> bool:
    if not settings.require_auth:
        return True
    auth = request.headers.get("authorization", "")
    expected = f"Bearer {settings.require_auth}"
    # constant-time-ish compare
    return auth == expected


@router.get("/v1/models")
async def list_models(request: Request):
    settings: Settings = request.app.state.settings
    if not _auth_ok(settings, request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return _model_list_payload(settings.models)


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    settings: Settings = request.app.state.settings
    if not _auth_ok(settings, request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="'messages' must be a non-empty list")

    # Concatenate text content for the classifier.
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    parts.append(str(c.get("text", "")))
    prompt = "\n".join(parts)

    # Transparent mode: if the client explicitly requested one of OUR tiers'
    # api ids, honor it directly instead of re-classifying.
    requested_model = body.get("model", "")
    known_tier = settings.models.tier_for_api_id(requested_model)
    if known_tier is not None:
        routed_tier = known_tier
    else:
        routed_tier = await classify(prompt, settings)

    routed_model = settings.models.tiers[routed_tier].api_id

    # Build the upstream body: swap the model id but keep everything else.
    upstream_body = dict(body)
    upstream_body["model"] = routed_model

    target_url = f"{settings.ollama_base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.effective_api_key}",
        "Content-Type": "application/json",
    }

    stream = bool(body.get("stream", False))
    upstream_body["stream"] = stream

    try:
        upstream = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=15.0))
        response = await upstream.post(target_url, headers=headers, json=upstream_body)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")

    if response.status_code >= 400:
        detail = response.text[:2000]
        await upstream.aclose()
        raise HTTPException(status_code=response.status_code, detail=detail)

    async def passthrough():
        if stream:
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()
        else:
            try:
                content = response.content
            finally:
                await upstream.aclose()
            yield content

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
