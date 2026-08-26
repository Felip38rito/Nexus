"""Tests for sync-opencode.py — the Nexus → OpenCode model sync.

Verifies the non-destructive guarantees that matter:
- Only the `router` provider's `models` VALUE is rewritten.
- Other providers and top-level keys are preserved byte-for-byte.
- Output is always valid JSON/JSONC.
- Re-running produces no further changes (idempotent).
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC_PATH = Path(__file__).parent.parent / "sync-opencode.py"
_SPEC = importlib.util.spec_from_file_location("sync_opencode", _SPEC_PATH)
assert _SPEC is not None and _SPEC.loader is not None
m = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(m)


SAMPLE = """{
  "$schema": "https://opencode.ai/config.json",
  "model": "router/adaptive",
  "provider": {
    "router": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Model Router (local)",
      "options": {
        "baseURL": "http://127.0.0.1:9000/v1",
        "apiKey": "router"
      },
      "models": {
        "adaptive": { "name": "Adaptive (auto-tier)", "limit": { "context": 1048576, "output": 65536 } },
        "old-model": { "name": "Stale (should disappear)", "limit": { "context": 1048576, "output": 65536 } }
      }
    },
    "other-provider": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Other",
      "models": {
        "keep-me": { "name": "Keep", "limit": { "context": 4096, "output": 2048 } }
      }
    }
  }
}
"""

MODELS = [
    {"id": "adaptive", "tier": "", "model": ""},
    {"id": "mini", "tier": "mini", "model": "gemma4:31b"},
    {"id": "air", "tier": "air", "model": "deepseek-v4-flash:0731"},
    {"id": "pro", "tier": "pro", "model": "deepseek-v4-pro:0813"},
    {"id": "ultra", "tier": "ultra", "model": "kimi-k3"},
]


def _run(text: str):
    return m.sync_config(text, MODELS)


def test_updates_models_and_removes_stale():
    new_text, changed, summary = _run(SAMPLE)
    assert changed is True
    assert "models block updated" in summary
    parsed = m.parse_jsonc(new_text)
    router = parsed["provider"]["router"]
    # Stale model removed, new tiers present.
    assert "old-model" not in router["models"]
    assert set(router["models"]) == {"adaptive", "mini", "air", "pro", "ultra"}
    # name/label for a tier reflects the upstream model id.
    assert router["models"]["mini"]["name"] == "Mini — gemma4:31b"


def test_preserves_other_providers_and_top_level():
    new_text, _, _ = _run(SAMPLE)
    parsed = m.parse_jsonc(new_text)
    assert parsed["$schema"] == "https://opencode.ai/config.json"
    assert parsed["model"] == "router/adaptive"
    other = parsed["provider"]["other-provider"]
    assert set(other["models"]) == {"keep-me"}
    assert other["models"]["keep-me"]["name"] == "Keep"
    # Router npm/name/options untouched.
    router = parsed["provider"]["router"]
    assert router["npm"] == "@ai-sdk/openai-compatible"
    assert router["options"]["baseURL"] == "http://127.0.0.1:9000/v1"


def test_output_is_valid_jsonc():
    new_text, _, _ = _run(SAMPLE)
    # parse_jsonc tolerates comments + trailing commas; must not raise.
    m.parse_jsonc(new_text)


def test_idempotent():
    once, _, _ = _run(SAMPLE)
    twice, changed, _ = _run(once)
    assert changed is False  # second run produces no further change


def test_adaptive_kept_and_tiered():
    new_text, _, _ = _run(SAMPLE)
    parsed = m.parse_jsonc(new_text)
    router = parsed["provider"]["router"]["models"]
    assert "adaptive" in router
    assert router["adaptive"]["name"] == "Adaptive (auto-tier)"
