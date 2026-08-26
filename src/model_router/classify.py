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

from .models import DEFAULT_TIER, RouterModels, Tier

log = logging.getLogger("model_router.classify")

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
