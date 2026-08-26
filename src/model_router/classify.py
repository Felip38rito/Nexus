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

import json
import logging
import re
from typing import Any

import httpx

from .models import DEFAULT_TIER, Tier

log = logging.getLogger("model_router.classify")

# Explicit model override wins over everything: "use deepseek-v4-pro", etc.
# This is the ONLY keyword-style match left — it targets specific model ids,
# not generic difficulty words, so it can't false-positive on normal prose.
_MODEL_OVERRIDE_RE = re.compile(
    r"\b(deepseek-v4-pro|deepseek-v4-flash|kimi-k3|gemma4:31b|glm-5\.2|minimax-m3)\b",
    re.IGNORECASE,
)


def _explicit_override(prompt: str) -> Tier | None:
    m = _MODEL_OVERRIDE_RE.search(prompt)
    if not m:
        return None
    token = m.group(1).lower()
    mapping = {
        "deepseek-v4-pro": Tier.PRO,
        "deepseek-v4-flash": Tier.AIR,
        "kimi-k3": Tier.ULTRA,
        "gemma4:31b": Tier.MINI,
    }
    # glm-5.2 / minimax-m3 are valid API ids but not routed tiers; treat as "ambiguous".
    if token in ("glm-5.2", "minimax-m3"):
        return None
    return mapping.get(token)


def deterministic_tier(prompt: str, min_classify_len: int = 10) -> Tier | None:
    """Decide only the obvious cases; return None to defer to the LLM.

    Two deterministic decisions, nothing more:

    1. Explicit model override — always wins, even for short prompts.
    2. Very short prompt (trivial chatter) — mini.

    Everything else returns None so the LLM makes the qualitative call. There
    are no difficulty keywords anymore: they were substring-matching common
    words ("hi" inside "this"/"crashing", "lock" inside "block"/"clock") and
    routing prompts to the wrong tier.
    """
    # 1. Explicit override wins over everything, including the length check.
    override = _explicit_override(prompt)
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
- mini   = trivial/mechanical/short chat, yes-no questions, simple lookups.
- air    = normal day-to-day implementation, straightforward coding, single-file edits.
- pro    = complex coding, hard debugging, refactors, multi-file impact, architecture analysis, codebase review, concurrency, public API design. Also: any request asking to "analyze", "evaluate", "assess", "review", or "improve" a project/codebase/system.
- ultra  = hardest problems, whole-architecture decisions, deep synthesis across many files, reverse-engineering, adversarial analysis, anything requiring maximum reasoning depth.

Think about what EXECUTING the request would require (reading many files, cross-referencing, deep reasoning) — not just the surface text. A short prompt like "analyze the whole project" implies reading dozens of files and synthesizing — that is pro or ultra, not mini.

Examples:
{"model": "mini", "reason": "simple greeting"}
{"model": "air", "reason": "straightforward single-file edit"}
{"model": "pro", "reason": "hard debugging, multi-file refactor needed"}
{"model": "pro", "reason": "avaliação de codebase inteira, reasoning complexo"}
{"model": "ultra", "reason": "whole-codebase refactor, deep synthesis required"}
{"model": "ultra", "reason": "revisar e melhorar o projeto, múltiplos módulos envolvidos"}"""


async def llm_tier(
    prompt: str,
    *,
    api_key: str,
    base_url: str = "https://ollama.com/v1",
    classifier_model: str = "gemma4:31b",
    client: httpx.AsyncClient | None = None,
) -> Tier | None:
    """Classify via a cheap LLM. Returns the tier or None on failure."""
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=30.0)
    try:
        body = {
            "model": classifier_model,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM},
                {"role": "user", "content": prompt[:4000]},
            ],
            "temperature": 0,
            "max_tokens": 120,
            "stream": False,
        }
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        # Extract the first {...} block defensively (models sometimes add prose).
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        parsed = json.loads(m.group(0)) if m else json.loads(text)
        candidate = str(parsed.get("model", "")).strip().lower()
        try:
            return Tier(candidate)
        except ValueError:
            log.warning("LLM classifier returned unknown tier %r", candidate)
            return None
    except Exception as exc:  # noqa: BLE001 - classify must never break the request
        log.warning("LLM classifier failed (%s); defaulting", exc)
        return None
    finally:
        if own_client:
            await client.aclose()


async def classify(
    prompt: str,
    settings: Any,
    client: httpx.AsyncClient | None = None,
) -> Tier:
    """Full hybrid path. Deterministic (override/trivial) first, LLM primary,
    default (air) last."""
    det = deterministic_tier(prompt, settings.min_classify_len)
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
