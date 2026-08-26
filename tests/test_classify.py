"""Tests for the deterministic + LLM-hybrid classifier."""
import json

import httpx
import pytest

from model_router.classify import deterministic_tier, llm_tier
from model_router.models import Tier


def test_trivial_short_prompt_mini():
    assert deterministic_tier("hi", min_classify_len=20) == Tier.MINI


def test_chat_default_air():
    # A normal-length question has no decisive keyword signal -> ambiguous (None).
    p = "What is the capital of France and what is its population?"
    assert deterministic_tier(p, min_classify_len=20) is None


def test_multistep_concurrency_pro():
    p = "Refactor this codebase to use @MainActor concurrency safely and write an ADR."
    assert deterministic_tier(p, min_classify_len=20) == Tier.PRO


def test_hard_debug_promax():
    p = "Please debug this stack trace in a legacy codebase and fix the race condition."
    assert deterministic_tier(p, min_classify_len=20) == Tier.PRO_MAX


def test_explicit_override_wins():
    assert deterministic_tier("use deepseek-v4-pro for this", min_classify_len=20) == Tier.PRO_MAX
    assert deterministic_tier("run gemma4:31b on this", min_classify_len=20) == Tier.MINI


def test_ambiguous_returns_none_for_llm():
    # Genuinely ambiguous — no decisive keyword signal -> None (LLM fallback).
    p = "Could you go over the changes on the landing page and share your overall thoughts on the layout?"
    assert deterministic_tier(p, min_classify_len=20) is None


def test_no_swift_only_bias():
    # A hard concurrency bug in Rust (not Swift) should still route to pro.
    p = "The goroutine scheduler deadlocks under load; fix the data race in the mutex-protected queue."
    assert deterministic_tier(p, min_classify_len=20) == Tier.PRO


def test_system_and_memory_signals_route_to_pro():
    p = "Investigate the segfault and memory leak in the C extension and improve latency."
    assert deterministic_tier(p, min_classify_len=20) == Tier.PRO


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
        return httpx.Response(500, text="boom")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tier = await llm_tier("some prompt", api_key="k", client=client)
    await client.aclose()
    assert tier is None
