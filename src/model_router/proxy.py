"""OpenAI-compatible relay: /v1/chat/completions + /v1/models.

The router accepts an OpenAI-format request, picks the cheapest adequate tier,
and streams the completion back from Ollama Cloud under that tier's model id.
"""
from __future__ import annotations

import hmac
import logging
import os
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .classify import classify
from .config import Settings
from .models import Tier

log = logging.getLogger("model_router.proxy")

router = APIRouter()


def _model_list_payload(models: "RouterModels") -> dict[str, Any]:
    """Advertise the virtual + tier model ids the router understands.

    The list IS the source of truth for what clients (e.g. the Hermes picker)
    may select. It exposes:
    - "adaptive"  -> the router decides the tier automatically.
    - mini/air/pro/ultra -> transparently force that tier.
    (The raw upstream api ids still work if sent directly, but they are not
    advertised so the picker stays clean and tier-oriented.)
    """
    data = [
        {
            "id": "adaptive",
            "object": "model",
            "created": 0,
            "owned_by": "model-router",
        }
    ]
    for tier in Tier:
        spec = models.tiers[tier]
        data.append(
            {
                "id": tier.value,
                "object": "model",
                "created": 0,
                "owned_by": "model-router",
                # "tier" metadata lets pickers group by tier without another probe.
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

    # Classify ONLY the LAST user message — the user's current intent. The
    # system prompt (Hermes' giant "You are Hermes Agent..." block) and
    # assistant/tool history are packed with pro/ultra keywords (analyze,
    # codebase, concurrency, lock, auth, performance, race condition, ...) that
    # would saturate the classifier. And concatenating ALL user messages from
    # the conversation means the classifier prompt grows over time, so a long
    # technical chat saturates the tier to pro/ultra even when the current
    # request is trivial. Using only the last user message keeps the tier
    # decision anchored to what the user is asking RIGHT NOW. Fall back to all
    # messages if there's no user message at all (defensive edge case).
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
        # Defensive: no user message at all — use any text we can find.
        parts: list[str] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
        prompt = "\n".join(parts)

    # Transparent mode: if the client explicitly requested one of OUR tiers'
    # api ids (raw upstream id OR a tier name like "mini"/"air"/"pro"/"ultra"),
    # honor it directly instead of re-classifying.
    requested_model = body.get("model", "")
    known_tier = settings.models.tier_for_api_id(requested_model)
    if known_tier is None:
        # Tier-name override ("mini", "air", "pro", "ultra") — force that tier.
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
    # Resolve which provider serves this tier (multi-provider support).
    provider = settings.models.provider_for(routed_spec.provider)

    # LOGGING (async-safe, via stdlib logging): terminal + rotating file.
    snippet = prompt[:50].replace("\n", " ") + "..."
    log.info(
        "Tier=%s Model=%s Provider=%s Prompt=%s",
        routed_tier.value,
        routed_model,
        provider.base_url,
        snippet,
    )

    # Build the upstream body: swap the model id but keep everything else.
    upstream_body = dict(body)
    upstream_body["model"] = routed_model

    target_url = f"{provider.base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ.get(provider.api_key_env, '')}",
        "Content-Type": "application/json",
    }

    stream = bool(body.get("stream", False))
    upstream_body["stream"] = stream

    # Shared client lives on app.state (created in lifespan). Fall back to a
    # one-off client for callers/tests that bypass the lifespan; that one-off
    # must be closed when the stream finishes (the shared client must NOT be).
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
