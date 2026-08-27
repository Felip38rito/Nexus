"""Hybrid router classifier.

The tier decision is fundamentally QUALITATIVE — it depends on intent, scope,
and context, not surface keywords. So the LLM is the PRIMARY decider, and the
deterministic layer is reduced to the two things it can do reliably:

1. An explicit model override in the prompt ("use deepseek-v4-pro") — always wins.
2. Obviously-trivial chatter (very short prompts) — routed to mini to save a
   round-trip.

Everything else is deferred to the LLM (`gemma4:31b`), which returns a strict
JSON decision. On any error the classifier degrades to the default tier (air) —
it never fails the request.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx

from .models import DEFAULT_TIER, RouterModels, Tier

log = logging.getLogger("model_router.classify")

# Classification is on the critical path of every request, so keep the upstream
# call fast: a short total timeout with a couple of quick retries on transient
# failures is better than letting each request hang for 30s before falling back.
_CLASSIFY_TIMEOUT = httpx.Timeout(10.0, connect=3.0)
# max_tokens must leave headroom for the JSON decision even if the model emits a
# little preamble before it; 120 was truncating the JSON and causing parse
# failures. 256 keeps the reply cheap but safe.
_CLASSIFY_MAX_TOKENS = 256
# Transient HTTP statuses worth retrying (server-side/upstream blips).
_RETRY_STATUS = {500, 502, 503, 504}
# Attempts total (1 initial + up to 2 retries).
_RETRY_ATTEMPTS = 3

# Explicit model override wins over everything: "use deepseek-v4-pro", etc.
# This is the ONLY keyword-style match left — it targets specific model ids,
# not generic difficulty words, so it can't false-positive on normal prose.
# The ids are built from the current model table so a YAML reconfiguration is
# honored automatically instead of drifting out of sync.
_OVERRIDE_WORDS_RE = re.compile(r"\b[a-z0-9_.:\-]+\b", re.IGNORECASE)


def _model_override(prompt: str, models: RouterModels) -> Tier | None:
    # Map each tier's api id -> tier. Users reference the bare model name
    # ("deepseek-v4-pro"), while the api id carries a version suffix
    # ("deepseek-v4-pro:0813"), so both the full id and its base are matched.
    # Only real model ids trigger an override — generic tier names ("pro"/"air")
    # are NOT matched here, so normal prose containing those words can't
    # accidentally pin a tier.
    lookup: dict[str, Tier] = {}
    for tier, spec in models.tiers.items():
        lookup[spec.api_id.lower()] = tier
        lookup[spec.api_id.split(":", 1)[0].lower()] = tier
    for token in _OVERRIDE_WORDS_RE.findall(prompt):
        tier = lookup.get(token.lower())
        if tier is not None:
            return tier
    return None


def deterministic_tier(prompt: str, min_classify_len: int = 10, models: RouterModels | None = None) -> Tier | None:
    """Decide only the obvious cases; return None to defer to the LLM.

    Two deterministic decisions, nothing more:

    1. Explicit model/tier override — always wins, even for short prompts.
    2. Very short prompt (trivial chatter) — mini.

    Everything else returns None so the LLM makes the qualitative call. There
    are no difficulty keywords anymore: they were substring-matching common
    words ("hi" inside "this"/"crashing", "lock" inside "block"/"clock") and
    routing prompts to the wrong tier.
    """
    models = models or RouterModels()

    # 1. Explicit override wins over everything, including the length check.
    override = _model_override(prompt, models)
    if override is not None:
        return override

    # 2. Trivial chatter: too short to be a real task.
    if len(prompt) < min_classify_len:
        return Tier.MINI

    # 3. Defer to the LLM for the qualitative decision.
    return None


LLM_SYSTEM = """You are a model router. Pick the most appropriate Ollama Cloud model tier for the user's request.

Reply with ONLY a single JSON object, no commentary, of the form {"model": "mini|air|pro|ultra", "reason": "<short>"}.

Tier definitions:
- mini   = general assistance, discussions, basic reasoning, simple lookups, and mechanical tasks.
- air    = the primary implementer. Handles almost all day-to-day coding, standard implementation, and multi-file edits when the path is clear.
- pro    = only for cases where air would struggle: deep debugging (race conditions, memory leaks), complex architectural refactors, high-level API design, or deep security audits.
- ultra  = maximum reasoning depth. Whole-system synthesis, adversarial analysis, reverse engineering, or solving "impossible" problems.

Think about what EXECUTING the request would require — not just the surface text. If it's a standard implementation task, it belongs in air. Only escalate to pro if the problem is inherently "hard" (e.g. a bug that requires deep state analysis), and to ultra if it requires synthesis of the entire system.

Examples:
{"model": "mini", "reason": "simple greeting"}
{"model": "air", "reason": "straightforward single-file edit"}
{"model": "pro", "reason": "hard debugging, multi-file refactor needed"}
{"model": "pro", "reason": "avaliação de codebase inteira, reasoning complexo"}
{"model": "ultra", "reason": "whole-codebase refactor, deep synthesis required"}
{"model": "ultra", "reason": "revisar e melhorar o projeto, múltiplos módulos envolvidos"}"""


def _extract_json_object(text: str) -> dict | None:
    """Robustly extract the first top-level JSON object from a model reply.

    Unlike a naive ``re.search(r"\\{.*?\\}", ...)`` (which breaks on a ``}``
    inside a string), this uses ``raw_decode`` from the first ``{`` to the
    *matching* closing brace, so embedded braces are handled correctly.
    """
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text, start)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


async def _classify_once(
    *,
    api_key: str,
    base_url: str,
    classifier_model: str,
    prompt: str,
    client: httpx.AsyncClient,
) -> httpx.Response:
    """Issue a single classifier request to the upstream."""
    body = {
        "model": classifier_model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": prompt[:4000]},
        ],
        "temperature": 0,
        "max_tokens": _CLASSIFY_MAX_TOKENS,
        "stream": False,
    }
    resp = await client.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=body,
    )
    return resp


def _parse_tier_response(resp: httpx.Response) -> Tier | None:
    """Parse a classifier HTTP response into a Tier, or None on any failure.

    Returns None (and logs a precise reason) if the body is missing, not JSON,
    empty, or contains no valid tier — the caller falls back to the default.
    """
    # Only accept a JSON content type; a 200 that came back as HTML/text means
    # the upstream returned an error page we can't trust.
    ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
    if ctype and ctype != "application/json":
        log.warning(
            "LLM classifier got non-JSON content-type %r (status %s); defaulting",
            ctype,
            resp.status_code,
        )
        return None

    raw = resp.text
    if not raw or not raw.strip():
        log.warning("LLM classifier returned empty body (status %s); defaulting", resp.status_code)
        return None

    try:
        data = resp.json()
    except json.JSONDecodeError:
        # Show a short body snippet so the failure isn't a mystery.
        snippet = raw[:200].replace("\n", " ")
        log.warning(
            "LLM classifier returned invalid JSON (status %s): %r; defaulting",
            resp.status_code,
            snippet,
        )
        return None

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        log.warning("LLM classifier response missing choices/message/content; defaulting")
        return None

    if not content or not str(content).strip():
        log.warning("LLM classifier returned empty content (status %s); defaulting", resp.status_code)
        return None

    obj = _extract_json_object(str(content))
    if obj is None:
        log.warning("LLM classifier reply had no parseable JSON object: %r; defaulting", str(content)[:200])
        return None

    candidate = str(obj.get("model", "")).strip().lower()
    try:
        return Tier(candidate)
    except ValueError:
        log.warning("LLM classifier returned unknown tier %r", candidate)
        return None


async def llm_tier(
    prompt: str,
    *,
    api_key: str,
    base_url: str = "https://ollama.com/v1",
    classifier_model: str = "gemma4:31b",
    client: httpx.AsyncClient | None = None,
) -> Tier | None:
    """Classify via a cheap LLM, with retry on transient failures.

    Returns the tier, or None on failure (the caller falls back to the default
    tier). Never raises — the router must not break a request just because the
    classifier is unavailable.
    """
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=_CLASSIFY_TIMEOUT)

    last_exc: Exception | None = None
    try:
        for attempt in range(_RETRY_ATTEMPTS):
            if attempt > 0:
                # Small exponential backoff with jitter (0.15s, 0.4s).
                delay = 0.1 * (2 ** (attempt - 1)) + (0.05 * attempt)
                await asyncio.sleep(delay)
            try:
                resp = await _classify_once(
                    api_key=api_key,
                    base_url=base_url,
                    classifier_model=classifier_model,
                    prompt=prompt,
                    client=client,
                )
                if resp.status_code in _RETRY_STATUS:
                    log.warning(
                        "LLM classifier upstream %s on attempt %d; retrying",
                        resp.status_code,
                        attempt + 1,
                    )
                    last_exc = httpx.HTTPStatusError(
                        f"status {resp.status_code}", request=resp.request, response=resp
                    )
                    continue
                resp.raise_for_status()
                tier = _parse_tier_response(resp)
                return tier  # None means parse failure -> fall back (no retry needed)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                # Network/connect blips are worth retrying.
                last_exc = exc
                log.warning("LLM classifier transport error on attempt %d: %s", attempt + 1, exc)
                continue
            except httpx.HTTPStatusError as exc:
                # Non-retryable status (e.g. 401/404/429) — give up immediately.
                last_exc = exc
                log.warning("LLM classifier HTTP %s (%s); defaulting", exc.response.status_code, exc)
                break
    finally:
        if own_client:
            await client.aclose()

    log.warning("LLM classifier failed (%s); defaulting", last_exc)
    return None


async def classify(
    prompt: str,
    settings: Any,
    client: httpx.AsyncClient | None = None,
) -> Tier:
    """Full hybrid path. Deterministic (override/trivial) first, LLM primary,
    default (air) last."""
    det = deterministic_tier(prompt, settings.min_classify_len, settings.models)
    if det is not None:
        return det
    llm = await llm_tier(
        prompt,
        api_key=settings.effective_api_key,
        base_url=settings.ollama_base_url,
        classifier_model=settings.models.classifier_model,
        client=client,
    )
    if llm is not None:
        return llm
    return settings.models.default_tier
