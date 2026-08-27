"""Tests for the LLM-primary classifier.

The deterministic layer now only handles two cases: explicit model overrides
and obviously-trivial chatter. Everything else defers to the LLM (returns None).
"""
import json

import httpx
import pytest

from model_router.classify import deterministic_tier, llm_tier
from model_router.models import Tier


def test_trivial_short_prompt_mini():
    assert deterministic_tier("hi", min_classify_len=10) == Tier.MINI


def test_short_greetings_mini():
    assert deterministic_tier("ok", min_classify_len=10) == Tier.MINI
    assert deterministic_tier("thanks", min_classify_len=10) == Tier.MINI
    assert deterministic_tier("fala", min_classify_len=10) == Tier.MINI


def test_explicit_override_wins():
    assert deterministic_tier("use deepseek-v4-pro for this", min_classify_len=10) == Tier.PRO
    assert deterministic_tier("run gemma4:31b on this", min_classify_len=10) == Tier.MINI
    assert deterministic_tier("use kimi-k3 for this", min_classify_len=10) == Tier.ULTRA


def test_override_wins_even_for_short_prompt():
    # Regression: the length check used to run BEFORE the override, so a short
    # prompt with an explicit model request was wrongly routed to mini.
    # "kimi-k3" is 7 chars (< min_classify_len) but the override must still win.
    assert deterministic_tier("kimi-k3", min_classify_len=10) == Tier.ULTRA


def test_non_routed_model_id_defers_to_llm():
    # glm-5.2 / minimax-m3 are still valid API ids but no longer routed tiers;
    # an explicit mention is treated as ambiguous and deferred to the LLM.
    assert deterministic_tier("use glm-5.2", min_classify_len=10) is None
    assert deterministic_tier("use minimax-m3", min_classify_len=10) is None


def test_short_technical_prompt_defers_to_llm():
    # "debug this segfault" is short but hard — must NOT be swallowed by the
    # length check. It defers to the LLM (None), which routes it to pro/ultra.
    assert deterministic_tier("debug this segfault", min_classify_len=10) is None


def test_normal_prompt_defers_to_llm():
    # No difficulty keywords anymore — a normal prompt is always deferred to the LLM.
    p = "What is the capital of France and what is its population?"
    assert deterministic_tier(p, min_classify_len=10) is None


def test_complex_prompt_defers_to_llm():
    # Even obviously-complex prompts defer to the LLM (qualitative decision).
    p = "Refactor this codebase to use @MainActor concurrency safely and write an ADR."
    assert deterministic_tier(p, min_classify_len=10) is None


def test_no_substring_false_positives():
    # Regression: "hi" used to substring-match inside "this"/"which"/"crashing",
    # dragging complex prompts down to mini. No keyword matching remains, so a
    # prompt containing "this"/"crashing" is simply deferred to the LLM.
    assert deterministic_tier("why is my app crashing", min_classify_len=10) is None
    assert deterministic_tier("review the architecture and suggest improvements", min_classify_len=10) is None


@pytest.mark.asyncio
async def test_llm_tier_maps_json():
    async def fake_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "gemma4:31b"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"model": "pro", "reason": "complex"}'}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(fake_handler))
    tier = await llm_tier("some long prompt", api_key="k", client=client)
    await client.aclose()
    assert tier == Tier.PRO


@pytest.mark.asyncio
async def test_llm_tier_handles_prose_around_json():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": 'Here you go: {"model": "air", "reason": "x"}'}}
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tier = await llm_tier("some prompt", api_key="k", client=client)
    await client.aclose()
    assert tier == Tier.AIR


@pytest.mark.asyncio
async def test_llm_tier_fails_safe_to_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tier = await llm_tier("some prompt", api_key="k", client=client)
    await client.aclose()
    assert tier is None


@pytest.mark.asyncio
async def test_llm_tier_retries_on_500_then_succeeds():
    """A 500 is transient — the classifier should retry and can succeed on a later attempt."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, text="boom", request=request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"model": "pro"}'}}]},
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tier = await llm_tier("some prompt", api_key="k", client=client)
    await client.aclose()
    assert tier == Tier.PRO
    assert calls["n"] == 2  # exactly one retry


@pytest.mark.asyncio
async def test_llm_tier_empty_body_returns_none():
    """A 200 with an empty body must not raise a JSONDecodeError — it falls back to None."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tier = await llm_tier("some prompt", api_key="k", client=client)
    await client.aclose()
    assert tier is None


@pytest.mark.asyncio
async def test_llm_tier_non_json_content_type_returns_none():
    """A 200 that came back as HTML/text (error page) must not be parsed as JSON."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"<html>oops</html>",
            headers={"content-type": "text/html"},
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tier = await llm_tier("some prompt", api_key="k", client=client)
    await client.aclose()
    assert tier is None


@pytest.mark.asyncio
async def test_llm_tier_empty_content_returns_none():
    """A 200 with valid JSON but empty content must fall back to None."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": ""}}]},
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tier = await llm_tier("some prompt", api_key="k", client=client)
    await client.aclose()
    assert tier is None


def test_extract_json_object_handles_braces_in_string():
    """A `}` inside a string value must not truncate the extracted object."""
    from model_router.classify import _extract_json_object

    text = 'Here: {"model": "air", "reason": "brace } inside", "x": 1} trailing'
    obj = _extract_json_object(text)
    assert obj is not None
    assert obj["model"] == "air"
    assert obj["reason"] == "brace } inside"
