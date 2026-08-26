"""OpenAI-compatible relay: /v1/chat/completions + /v1/models.

The router accepts an OpenAI-format request, picks the cheapest adequate tier,
and streams the completion back from Ollama Cloud under that tier's model id.
"""
from __future__ import annotations

from typing import Any
import datetime
import httpx
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .classify import classify
from .config import Settings

router = APIRouter()


def _model_list_payload(models: "RouterModels") -> dict[str, Any]:
    # "adaptive" is a virtual model id — always triggers classification.
    data = [
        {
            "id": "adaptive",
            "object": "model",
            "created": 0,
            "owned_by": "model-router",
        }
    ]
    for spec in models.tiers.values():
        data.append(
            {
                "id": spec.api_id,
                "object": "model",
                "created": 0,
                "owned_by": "ollama-cloud",
            }
        )
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
    # api ids, honor it directly instead of re-classifying.
    requested_model = body.get("model", "")
    known_tier = settings.models.tier_for_api_id(requested_model)
    if known_tier is not None:
        routed_tier = known_tier
    else:
        routed_tier = await classify(prompt, settings)

    routed_model = settings.models.tiers[routed_tier].api_id

    # LOGGING: Print to terminal first (guaranteed) then save to file
    timestamp = datetime.datetime.now().isoformat()
    snippet = prompt[:50].replace("\n", " ") + "..."
    
    # Print to terminal immediately
    print(f"\n📡 [Router] {routed_tier.value} ➔ {routed_model} | Prompt: {snippet}\n", flush=True)

    try:
        log_path = Path(settings.project_root) / "router.log"
        log_entry = f"{timestamp} | Prompt: {snippet} | Tier: {routed_tier.value} | Model: {routed_model}\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception:
        pass # Don't break the proxy if file logging fails

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
