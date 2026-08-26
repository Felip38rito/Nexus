"""Hybrid router classifier.

1. Deterministic keyword rules run first (free, ~instant). If the evidence is
   clear enough, a tier is returned.
2. If the rules are ambiguous (conflicting / no signal), fall back to a cheap
   LLM call (`gemma4:31b`) that returns a strict JSON decision.
3. On any parse/network error the classifier degrades to the default tier —
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
_MODEL_OVERRIDE_RE = re.compile(
    r"\b(deepseek-v4-pro|deepseek-v4-flash|glm-5\.2|gemma4:31b|minimax-m3|kimi-k3)\b",
    re.IGNORECASE,
)

# Weighted keyword signals. Scoring is additive; ties/emptiness -> None (LLM).
_PRO_MAX_KEYWORDS = [
    "pro-max", "pro max", "deepseek-v4-pro", "swe-bench",
    "raw coding", "adversarial", "reverse-engineer",
    "refactor this entire", "rewrite the whole", "legacy codebase", "bytecode",
    "compiler", "linker", "kernel module", "race condition", "exploit", "pwn",
]
# Complexity signals — deliberately language-agnostic (Swift, Rust, Go, Python,
# JS/TS, Java, C/C++, systems). These flag hard technical work, not any one stack.
_PRO_KEYWORDS = [
    # architecture / design
    "architecture decision", "adr", "ambiguous", "trade-off", "tradeoff",
    "non-trivial", "non trivial", "architectural", "design decision",
    # concurrency & threading (multi-language)
    "concurrency", "multithreading", "thread-safe", "thread safety",
    "async", "await", "goroutine", "mutex", "lock", "deadlock",
    "race condition", "data race", "actor", "actor model", "isolate",
    # memory / performance / safety
    "memory leak", "segfault", "segmentation fault", "unsafe", "borrow",
    "garbage collector", "gc pause", "crash", "stack trace", "buffer overflow",
    "out of memory", "performance", "latency",
    # debugging
    "debug", "debugging", "hard to reproduce", "regression",
    # api / systems / security
    "public api", "sdk", "protocol", "distributed", "kernel", "migration",
    "encryption", "auth", "authentication", "authorization",
]
# Words that bump things DOWN (common in chat / trivial asks).
_MIN_KEYWORDS = [
    "rename", "typo", "format", "reformat", "spell-check", "rephrase",
    "summarize", "transcribe", "what is", "who is", "explain briefly",
    "simple", "trivial", "quick", "boilerplate", "hello", "hi", "thanks",
]


def _score(prompt: str, keywords: list[str]) -> int:
    low = prompt.lower()
    return sum(1 for kw in keywords if kw.lower() in low)


def _explicit_override(prompt: str) -> Tier | None:
    m = _MODEL_OVERRIDE_RE.search(prompt)
    if not m:
        return None
    token = m.group(1).lower()
    mapping = {
        "deepseek-v4-pro": Tier.PRO_MAX,
        "deepseek-v4-flash": Tier.AIR,
        "glm-5.2": Tier.PRO,
        "gemma4:31b": Tier.MINI,
    }
    # kimi / minimax are valid API ids but not routed tiers; treat as "ambiguous".
    if token in ("minimax-m3", "kimi-k3"):
        return None
    return mapping.get(token)


def deterministic_tier(prompt: str, min_classify_len: int = 80) -> Tier | None:
    """Return a tier from keyword evidence, or None if ambiguous."""
    if len(prompt) < min_classify_len:
        return Tier.MINI

    override = _explicit_override(prompt)
    if override is not None:
        return override

    pro_max_score = _score(prompt, _PRO_MAX_KEYWORDS)
    pro_score = _score(prompt, _PRO_KEYWORDS)
    mini_score = _score(prompt, _MIN_KEYWORDS)

    # Heavy signals dominate.
    if pro_max_score >= 1:
        return Tier.PRO_MAX
    if pro_score >= 2:
        return Tier.PRO
    # A single pro keyword plus no down-signal is weakly pro.
    if pro_score == 1 and mini_score == 0:
        return Tier.PRO
    # Nothing but chatter.
    if mini_score >= 2:
        return Tier.MINI
    return None


LLM_SYSTEM = (
    "You are a model router. Decide the cheapest adequate Ollama Cloud model tier "
    "for the user's request. Reply with ONLY a single JSON object, no commentary, "
    "of the form {\"model\": \"mini|air|pro|pro-max\", \"reason\": \"<short>\"}. "
    "Rules: mini = trivial/mechanical/short chat; air = normal day-to-day; "
    "pro = complex reasoning, ambiguous design, concurrency, public API; "
    "pro-max = hardest debugging/refactors/architecture."
)


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
            "max_tokens": 60,
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
    """Full hybrid path. Deterministic first, LLM fallback, default last."""
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
