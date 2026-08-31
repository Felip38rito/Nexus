"""Tests for config + model tier table integrity."""
import os
from pathlib import Path

import pytest

from model_router.config import Settings, load_models_yaml
from model_router.models import MODEL_TABLE, Tier, tier_for_api_id, ModelSpec


def test_tier_table_has_all_tiers():
    assert set(MODEL_TABLE.keys()) == {Tier.MINI, Tier.AIR, Tier.PRO, Tier.ULTRA}


def test_api_ids_are_not_cloud_suffixed():
    # The :cloud suffix is an Hermes alias only; the API requires raw ids.
    for spec in MODEL_TABLE.values():
        assert not spec.api_id.endswith(":cloud")


def test_tier_for_api_id_roundtrip():
    for tier, spec in MODEL_TABLE.items():
        assert tier_for_api_id(spec.api_id) == tier
    assert tier_for_api_id("glm-5.1") is None


def test_settings_from_env(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "OLLAMA_API_KEY=secret\nROUTER_PORT=9999\nROUTER_DEFAULT_TIER=pro\n"
    )
    s = Settings.from_env(env)
    assert s.ollama_api_key == "secret"
    assert s.router_port == 9999
    assert s.default_tier == Tier.PRO


def test_settings_effective_key_raises_when_missing(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("ROUTER_PORT=8000\n")
    s = Settings.from_env(env)
    with pytest.raises(RuntimeError):
        _ = s.effective_api_key


def _sample_yaml() -> str:
    return """\
default_tier: air
tiers:
  mini:
    model: gemma4:31b
    description: "fast"
  air:
    model: deepseek-v4-flash:0731
    description: "default"
  pro:
    model: deepseek-v4-pro:0813
    description: "complex"
  ultra:
    model: kimi-k3
    description: "hard"
classifier:
  model: gemma4:31b
  min_classify_len: 40
"""


def test_load_models_yaml_custom(tmp_path: Path):
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(_sample_yaml())
    models = load_models_yaml(yaml_path)
    assert models is not None
    assert models.tiers[Tier.MINI].api_id == "gemma4:31b"
    assert models.classifier_model == "gemma4:31b"
    assert models.min_classify_len == 40
    assert models.default_tier == Tier.AIR


def test_load_models_yaml_missing_returns_none(tmp_path: Path):
    assert load_models_yaml(tmp_path / "nope.yaml") is None


def test_load_models_yaml_unknown_tier_raises(tmp_path: Path):
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text(
        "tiers:\n  bogus:\n    model: x\n  air:\n    model: y\n"
        "  pro:\n    model: z\n  ultra:\n    model: w\n  mini:\n    model: m\n"
    )
    with pytest.raises(ValueError, match="Unknown tier"):
        load_models_yaml(yaml_path)


def test_load_models_yaml_missing_tier_raises(tmp_path: Path):
    yaml_path = tmp_path / "bad2.yaml"
    # Omit the "pro" tier entirely.
    yaml_path.write_text(
        "tiers:\n  air:\n    model: y\n  ultra:\n    model: w\n  mini:\n    model: m\n"
    )
    with pytest.raises(ValueError, match="Missing tier"):
        load_models_yaml(yaml_path)


def test_settings_from_env_loads_yaml(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(_sample_yaml())
    env = tmp_path / ".env"
    env.write_text("OLLAMA_API_KEY=secret\n")
    s = Settings.from_env(env, default_models_yaml=yaml_path)
    assert s.models.min_classify_len == 40
    assert s.models.tiers[Tier.PRO].api_id == "deepseek-v4-pro:0813"
    assert s.models.tiers[Tier.ULTRA].api_id == "kimi-k3"


def _multi_provider_yaml() -> str:
    return """\
default_tier: air
providers:
  default:
    base_url: https://ollama.com/v1
    api_key_env: OLLAMA_API_KEY
  gemini:
    base_url: https://generativelanguage.googleapis.com/v1beta
    api_key_env: GEMINI_API_KEY
tiers:
  mini:
    model: gemma4:31b
    description: "fast"
  air:
    model: deepseek-v4-flash:0731
    description: "default"
  pro:
    model: gemini-2.5-pro
    description: "complex"
    provider: gemini
  ultra:
    model: kimi-k3
    description: "hard"
classifier:
  model: gemma4:31b
  provider: default
  min_classify_len: 40
"""


def test_load_models_yaml_multi_provider(tmp_path: Path):
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(_multi_provider_yaml())
    models = load_models_yaml(yaml_path)
    assert models is not None
    # Tiers default to "default" provider unless overridden.
    assert models.tiers[Tier.MINI].provider == "default"
    assert models.tiers[Tier.AIR].provider == "default"
    # The pro tier points at the gemini provider.
    assert models.tiers[Tier.PRO].provider == "gemini"
    assert models.tiers[Tier.PRO].api_id == "gemini-2.5-pro"
    # Classifier provider resolved.
    assert models.classifier_provider == "default"
    # Providers table populated.
    assert "gemini" in models.providers
    assert models.providers["gemini"].base_url == "https://generativelanguage.googleapis.com/v1beta"
    assert models.providers["gemini"].api_key_env == "GEMINI_API_KEY"
    # provider_for resolves both.
    assert models.provider_for("gemini").api_key_env == "GEMINI_API_KEY"
    assert models.provider_for("default").base_url == "https://ollama.com/v1"


def test_load_models_yaml_unknown_provider_raises(tmp_path: Path):
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text(
        "tiers:\n"
        "  mini:\n    model: m\n    provider: nope\n"
        "  air:\n    model: y\n"
        "  pro:\n    model: z\n"
        "  ultra:\n    model: w\n"
    )
    with pytest.raises(ValueError, match="unknown provider 'nope'"):
        load_models_yaml(yaml_path)


def test_load_models_yaml_provider_requires_base_url(tmp_path: Path):
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text(
        "providers:\n"
        "  broken:\n    api_key_env: X\n"
        "tiers:\n"
        "  mini:\n    model: m\n"
        "  air:\n    model: y\n"
        "  pro:\n    model: z\n"
        "  ultra:\n    model: w\n"
    )
    with pytest.raises(ValueError, match="must define a 'base_url'"):
        load_models_yaml(yaml_path)


def test_modelspec_name_defaults_to_none():
    from model_router.models import ModelSpec
    spec = ModelSpec(api_id="gemma4:31b", description="x")
    assert spec.name is None
