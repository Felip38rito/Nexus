"""Tests for the axon setup flow (config.yml generation)."""
from __future__ import annotations

from typing import Any

import pytest

from axon_cli import setup
from model_router.models import Tier


# --- _build_config (pure) ---------------------------------------------------

def _tiers(models: dict[Tier, str] | None = None) -> dict[Tier, dict[str, Any]]:
    """Build a minimal tiers dict with the given model ids (defaults to examples)."""
    models = models or setup.EXAMPLE_MODELS
    return {
        tier: {"name": tier.value, "model": model}
        for tier, model in models.items()
    }


def test_build_config_env_var_provider():
    provider = {
        "name": "default",
        "base_url": "https://ollama.com/v1",
        "api_key_env": "OLLAMA_API_KEY",
    }
    config = setup._build_config(provider, _tiers())

    assert config["default_tier"] == "air"
    assert config["providers"]["default"]["base_url"] == "https://ollama.com/v1"
    assert config["providers"]["default"]["api_key_env"] == "OLLAMA_API_KEY"
    # No inline key should leak into the env-var path.
    assert "api_key" not in config["providers"]["default"]
    # All four tiers present with correct model ids.
    assert config["tiers"]["mini"]["model"] == "gemma4:31b"
    assert config["tiers"]["pro"]["model"] == "deepseek-v4-pro:0813"
    # Classifier points at the provider.
    assert config["classifier"]["provider"] == "default"
    assert config["classifier"]["model"] == "gemma4:31b"


def test_build_config_inline_key_provider():
    provider = {
        "name": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-inline-secret",
    }
    config = setup._build_config(provider, _tiers())

    assert config["providers"]["openai"]["api_key"] == "sk-inline-secret"
    # Inline key must NOT also emit an api_key_env (avoids ambiguity).
    assert "api_key_env" not in config["providers"]["openai"]
    assert config["classifier"]["provider"] == "openai"


def test_build_config_extra_params():
    tiers = _tiers()
    tiers[Tier.AIR]["extra_params"] = {"reasoning_effort": "high"}
    config = setup._build_config(
        {"name": "default", "base_url": "https://ollama.com/v1", "api_key_env": "X"},
        tiers,
    )
    assert config["tiers"]["air"]["extra_params"] == {"reasoning_effort": "high"}
    # Tiers without extra_params omit the key entirely.
    assert "extra_params" not in config["tiers"]["mini"]


# --- _required_prompt -------------------------------------------------------

def test_required_prompt_rejects_blank(monkeypatch):
    calls = iter(["", "   ", "real-value"])
    monkeypatch.setattr(setup.Prompt, "ask", lambda prompt: next(calls))
    assert setup._required_prompt("anything") == "real-value"


def test_required_prompt_strips_whitespace(monkeypatch):
    monkeypatch.setattr(setup.Prompt, "ask", lambda prompt: "  padded  ")
    assert setup._required_prompt("anything") == "padded"


# --- _prompt_provider -------------------------------------------------------

def test_prompt_provider_env_var(monkeypatch):
    answers = iter(["default", "https://ollama.com/v1", "OLLAMA_API_KEY"])
    monkeypatch.setattr(setup.Prompt, "ask", lambda prompt: next(answers))
    monkeypatch.setattr(setup.Confirm, "ask", lambda *a, **k: False)

    result = setup._prompt_provider()
    assert result == {
        "name": "default",
        "base_url": "https://ollama.com/v1",
        "api_key_env": "OLLAMA_API_KEY",
    }


def test_prompt_provider_inline_key(monkeypatch):
    answers = iter(["openai", "https://api.openai.com/v1", "sk-secret"])
    monkeypatch.setattr(setup.Prompt, "ask", lambda prompt: next(answers))
    monkeypatch.setattr(setup.Confirm, "ask", lambda *a, **k: True)

    result = setup._prompt_provider()
    assert result == {
        "name": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-secret",
    }
    assert "api_key_env" not in result


def test_prompt_provider_strips_trailing_slash(monkeypatch):
    answers = iter(["default", "https://ollama.com/v1/", "OLLAMA_API_KEY"])
    monkeypatch.setattr(setup.Prompt, "ask", lambda prompt: next(answers))
    monkeypatch.setattr(setup.Confirm, "ask", lambda *a, **k: False)

    result = setup._prompt_provider()
    assert result["base_url"] == "https://ollama.com/v1"


# --- _prompt_tiers ----------------------------------------------------------

def test_prompt_tiers_omakase(monkeypatch):
    # Omakase: names come from tier keys; only model ids are prompted.
    model_answers = iter(setup.EXAMPLE_MODELS.values())
    monkeypatch.setattr(setup.Confirm, "ask", lambda *a, **k: True)
    monkeypatch.setattr(
        setup.Prompt,
        "ask",
        lambda prompt: next(model_answers),
    )

    tiers = setup._prompt_tiers()
    assert tiers[Tier.MINI]["name"] == "mini"
    assert tiers[Tier.MINI]["model"] == "gemma4:31b"
    assert tiers[Tier.ULTRA]["model"] == "kimi-k3"


def test_prompt_tiers_custom_names(monkeypatch):
    # Custom: display name is prompted (defaults to tier key), then model id.
    answers = iter([
        "Fast", "gemma4:31b",
        "Daily", "deepseek-v4-flash:0731",
        "Power", "deepseek-v4-pro:0813",
        "Deep", "kimi-k3",
    ])
    monkeypatch.setattr(setup.Confirm, "ask", lambda *a, **k: False)
    monkeypatch.setattr(setup.Prompt, "ask", lambda prompt, **k: next(answers))

    tiers = setup._prompt_tiers()
    assert tiers[Tier.MINI]["name"] == "Fast"
    assert tiers[Tier.PRO]["name"] == "Power"
    assert tiers[Tier.PRO]["model"] == "deepseek-v4-pro:0813"


# --- _prompt_reasoning ------------------------------------------------------

def test_prompt_reasoning_skipped_when_declined(monkeypatch):
    tiers = _tiers()
    monkeypatch.setattr(setup.Confirm, "ask", lambda *a, **k: False)
    setup._prompt_reasoning(tiers)
    # No tier gained extra_params.
    assert all("extra_params" not in t for t in tiers.values())


def test_prompt_reasoning_adds_params(monkeypatch):
    tiers = _tiers()
    # First Confirm (enable reasoning) -> True; then tier loop.
    confirm_answers = iter([True])
    monkeypatch.setattr(setup.Confirm, "ask", lambda *a, **k: next(confirm_answers))
    # Prompt.ask sequence: tier key, param key, param value, done, done.
    prompt_answers = iter(["air", "reasoning_effort", "high", "done", "done"])
    monkeypatch.setattr(setup.Prompt, "ask", lambda prompt, **k: next(prompt_answers))

    setup._prompt_reasoning(tiers)
    assert tiers[Tier.AIR]["extra_params"] == {"reasoning_effort": "high"}
    assert "extra_params" not in tiers[Tier.MINI]


def test_prompt_reasoning_unknown_tier_loops(monkeypatch):
    tiers = _tiers()
    confirm_answers = iter([True])
    monkeypatch.setattr(setup.Confirm, "ask", lambda *a, **k: next(confirm_answers))
    # First tier key is bogus (loops), then 'done' exits.
    prompt_answers = iter(["bogus", "done"])
    monkeypatch.setattr(setup.Prompt, "ask", lambda prompt, **k: next(prompt_answers))

    setup._prompt_reasoning(tiers)
    assert all("extra_params" not in t for t in tiers.values())


# --- run_setup --------------------------------------------------------------

def test_run_setup_writes_config(monkeypatch, tmp_path):
    """run_setup writes a valid config.yml to the config path."""
    target = tmp_path / "config.yml"
    monkeypatch.setattr(setup, "config_path", lambda: target)

    # _prompt_provider: name, base_url, then Confirm(inline)=False, then api_key_env.
    provider_answers = iter(["default", "https://ollama.com/v1", "OLLAMA_API_KEY"])
    # _prompt_tiers: Confirm(omakase)=True, then 4 model ids.
    tier_answers = iter(setup.EXAMPLE_MODELS.values())
    # _prompt_reasoning: Confirm(enable)=False.
    confirm_answers = iter([False, True, False])

    def fake_confirm(*a, **k):
        return next(confirm_answers)

    def fake_prompt(prompt, **k):
        # Route to the right answer stream based on the prompt text.
        if "Provider name" in prompt or "base_url" in prompt or "API key" in prompt:
            return next(provider_answers)
        return next(tier_answers)

    monkeypatch.setattr(setup.Confirm, "ask", fake_confirm)
    monkeypatch.setattr(setup.Prompt, "ask", fake_prompt)

    setup.run_setup()

    import yaml as _yaml
    data = _yaml.safe_load(target.read_text())
    assert data["providers"]["default"]["api_key_env"] == "OLLAMA_API_KEY"
    assert data["tiers"]["mini"]["model"] == "gemma4:31b"
    assert data["tiers"]["ultra"]["model"] == "kimi-k3"
    assert data["classifier"]["provider"] == "default"
