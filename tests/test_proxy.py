"""Tests for the OpenAI-compatible proxy endpoints."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from model_router.config import Settings
from model_router.main import create_app


def _make_app(settings: Settings | None = None):
    return create_app(settings or Settings.from_env(None))


@pytest.fixture
def client():
    settings = Settings(
        ollama_api_key="upstream-key",
        ollama_base_url="https://ollama.com/v1",
    )
    return TestClient(_make_app(settings))


def test_list_models(client: TestClient):
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    ids = {m["id"] for m in data}
    # The router advertises the tier names (source of truth for pickers), not
    # the raw upstream api ids.
    assert ids == {"adaptive", "mini", "air", "pro", "ultra"}
    # Each tier row also carries the upstream model + tier metadata.
    by_id = {m["id"]: m for m in data}
    assert by_id["mini"]["model"] == "gemma4:31b"
    assert by_id["air"]["model"] == "deepseek-v4-flash:0731"
    assert by_id["pro"]["tier"] == "pro"


def test_adaptive_always_classifies(client: TestClient, monkeypatch):
    """'adaptive' is a virtual id — must always trigger classification, never passthrough."""
    seen = {}

    async def fake_post(self, url, headers, **kw):
        json_body = kw["json"]
        seen["model"] = json_body["model"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    # Trivial prompt with model="adaptive" should classify to mini, not passthrough.
    payload = {
        "model": "adaptive",
        "messages": [{"role": "user", "content": "hi"}],
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    assert seen["model"] == "gemma4:31b"
    assert r.headers["X-Router-Tier"] == "mini"


def test_chat_completion_routes_and_streams(client: TestClient, monkeypatch):
    seen = {}

    async def fake_post(self, url, headers=None, **kw):
        json_body = kw["json"]
        seen["url"] = url
        seen["model"] = json_body["model"]
        seen["stream"] = json_body["stream"]
        data = [
            {"id": "1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "hi"}}]},
            {"id": "1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}}]},
        ]
        body = "\n".join("data: " + json.dumps(d) for d in data) + "\ndata: [DONE]\n\n"
        return httpx.Response(200, content=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    payload = {
        "model": "any-routed-model",
        "messages": [{"role": "user", "content": "what is the capital of france?"}],
        "stream": True,
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    # routed model header set (this prompt is ambiguous -> LLM fallback would
    # be invoked, but we stub post so it returns None -> default air)
    assert r.headers["X-Router-Model"] in {"gemma4:31b", "deepseek-v4-flash:0731", "deepseek-v4-pro:0813", "kimi-k3"}
    assert "text/event-stream" in r.headers["content-type"]
    assert "data: " in r.text


def test_chat_completion_trivial_routes_to_mini(client: TestClient, monkeypatch):
    seen = {}

    async def fake_post(self, url, headers, **kw):
        json_body = kw["json"]
        seen["model"] = json_body["model"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    payload = {
        "model": "anything",
        "messages": [{"role": "user", "content": "hi"}],
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    assert seen["model"] == "gemma4:31b"
    assert r.headers["X-Router-Tier"] == "mini"


def test_system_prompt_keywords_do_not_poison_classification(client: TestClient, monkeypatch):
    """The Hermes system prompt is full of pro/ultra keywords (analyze,
    codebase, concurrency, lock, auth, performance, race condition, ...). It must
    NOT feed the classifier — only the user's intent should drive the tier.
    Regression: previously ALL messages were concatenated, so every request with
    the Hermes system prompt saturated the deterministic rules and routed to pro."""
    seen = {}

    async def fake_post(self, url, headers, **kw):
        json_body = kw["json"]
        seen["model"] = json_body["model"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    system_prompt = (
        "You are Hermes Agent. You assist with analyzing information, writing and "
        "editing code, concurrency, lock-free queues, auth, performance and latency "
        "analysis, kernel modules, race conditions, codebase reviews, memory leaks, "
        "and running git commands. Analyze, evaluate and assess. Use lock and auth. "
    )
    payload = {
        "model": "anything",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "fala aliado, na escuta?"},
        ],
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    # The trivial user message must NOT be dragged up to pro by system keywords.
    # "fala aliado, na escuta?" is ambiguous (len>min_classify_len) -> LLM fallback
    # (fails on stub) -> default air. Crucially it must NOT be pro/ultra.
    assert r.headers["X-Router-Tier"] in {"air", "mini"}
    assert seen["model"] in {"deepseek-v4-flash:0731", "gemma4:31b"}


def test_long_history_does_not_saturate_tier(client: TestClient, monkeypatch):
    """Regression: concatenating ALL user messages means a long technical
    conversation grows the classifier prompt, so even a trivial follow-up
    ("obrigado!") gets routed to pro/ultra. The fix: classify ONLY the last
    user message. This test sends a conversation with several complex user
    messages followed by a trivial last message — the router must NOT route to
    pro/ultra."""
    seen = {}

    async def fake_post(self, url, headers, **kw):
        json_body = kw["json"]
        seen["model"] = json_body["model"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    messages = [
        {"role": "system", "content": "You are Hermes Agent. Analyze codebase, concurrency, race conditions."},
        {"role": "user", "content": "Analyze the whole project architecture and evaluate the concurrency model"},
        {"role": "assistant", "content": "I analyzed the codebase..."},
        {"role": "user", "content": "Now review the kernel module and assess the lock-free queue performance"},
        {"role": "assistant", "content": "The race condition in..."},
        {"role": "user", "content": "valeu!"},  # trivial last message
    ]
    payload = {"model": "adaptive", "messages": messages}
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    # "valeu!" is < min_classify_len (10) → deterministic mini.
    assert r.headers["X-Router-Tier"] == "mini"
    assert seen["model"] == "gemma4:31b"


def test_transparent_known_model_passthrough(client: TestClient, monkeypatch):
    seen = {}

    async def fake_post(self, url, headers, **kw):
        json_body = kw["json"]
        seen["model"] = json_body["model"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    payload = {
        "model": "deepseek-v4-pro:0813",  # explicit known api id
        "messages": [{"role": "user", "content": "hi"}],
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    assert seen["model"] == "deepseek-v4-pro:0813"


def test_invalid_messages_rejected(client: TestClient):
    r = client.post("/v1/chat/completions", json={"model": "x", "messages": []})
    assert r.status_code == 400


def test_upstream_error_maps_to_status(client: TestClient, monkeypatch):
    async def fake_post(self, url, headers, **kw):
        return httpx.Response(500, text="upstream boom", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    payload = {
        "model": "x",
        "messages": [{"role": "user", "content": "hi"}],
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 500


def test_optional_auth(client: TestClient):
    r = client.get("/v1/models")
    assert r.status_code == 200


def test_required_auth_enforced():
    settings = Settings(ollama_api_key="k", require_auth="router-secret")
    client = TestClient(_make_app(settings))
    assert client.get("/v1/models").status_code == 401
    ok = client.get("/v1/models", headers={"Authorization": "Bearer router-secret"})
    assert ok.status_code == 200


def test_custom_models_yaml_drives_proxy(tmp_path, monkeypatch):
    """A custom models YAML changes both /v1/models and the routed model."""
    from model_router.config import load_models_yaml

    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text(
        "default_tier: air\n"
        "tiers:\n"
        "  mini:\n"
        "    model: my-mini\n"
        "    description: \"cheap\"\n"
        "  air:\n"
        "    model: my-air\n"
        "    description: \"default\"\n"
        "  pro:\n"
        "    model: my-pro\n"
        "    description: \"reasoning\"\n"
        "  ultra:\n"
        "    model: my-ultra\n"
        "    description: \"hard\"\n"
        "classifier:\n"
        "  model: my-classifier\n"
        "  min_classify_len: 5\n"
    )
    models = load_models_yaml(yaml_path)
    settings = Settings(
        ollama_api_key="upstream-key",
        ollama_base_url="https://ollama.com/v1",
        models=models,
    )
    client = TestClient(_make_app(settings))

    # /v1/models always advertises the tier names (source of truth); the
    # custom upstream ids are exposed via each row's "model" metadata.
    data = client.get("/v1/models").json()["data"]
    ids = {m["id"] for m in data}
    assert ids == {"adaptive", "mini", "air", "pro", "ultra"}
    by_id = {m["id"]: m for m in data}
    assert by_id["mini"]["model"] == "my-mini"
    assert by_id["ultra"]["model"] == "my-ultra"

    # A trivial prompt routes to the custom mini model.
    seen = {}

    async def fake_post(self, url, headers, **kw):
        seen["model"] = kw["json"]["model"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "whatever", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert seen["model"] == "my-mini"
    assert r.headers["X-Router-Tier"] == "mini"
    assert r.headers["X-Router-Model"] == "my-mini"


def test_multi_provider_routes_to_correct_endpoint(tmp_path, monkeypatch):
    """A tier pointing at a non-default provider must hit that provider's URL."""
    from model_router.config import load_models_yaml

    yaml_path = tmp_path / "multi.yaml"
    yaml_path.write_text(
        "default_tier: air\n"
        "providers:\n"
        "  default:\n"
        "    base_url: https://ollama.com/v1\n"
        "    api_key_env: OLLAMA_API_KEY\n"
        "  gemini:\n"
        "    base_url: https://generativelanguage.googleapis.com/v1beta\n"
        "    api_key_env: GEMINI_API_KEY\n"
        "tiers:\n"
        "  mini:\n"
        "    model: gemma4:31b\n"
        "  air:\n"
        "    model: deepseek-v4-flash:0731\n"
        "  pro:\n"
        "    model: gemini-2.5-pro\n"
        "    provider: gemini\n"
        "  ultra:\n"
        "    model: kimi-k3\n"
        "classifier:\n"
        "  model: gemma4:31b\n"
        "  min_classify_len: 5\n"
    )
    models = load_models_yaml(yaml_path)
    settings = Settings(
        ollama_api_key="upstream-key",
        ollama_base_url="https://ollama.com/v1",
        models=models,
    )
    client = TestClient(_make_app(settings))

    seen = {}

    async def fake_post(self, url, headers, **kw):
        seen["url"] = url
        seen["model"] = kw["json"]["model"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    # Force the pro tier (which points at gemini) via a tier-name override.
    r = client.post(
        "/v1/chat/completions",
        json={"model": "pro", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert seen["model"] == "gemini-2.5-pro"
    # The request must go to the gemini provider's endpoint, not ollama.
    assert seen["url"] == "https://generativelanguage.googleapis.com/v1beta/chat/completions"
    assert r.headers["X-Router-Tier"] == "pro"

def test_models_payload_uses_custom_name():
    from model_router.models import RouterModels, ModelSpec, Tier
    from model_router.proxy import _model_list_payload

    models = RouterModels(
        tiers={
            Tier.MINI: ModelSpec("gemma4:31b", "d", name="Fast"),
            Tier.AIR: ModelSpec("deepseek-v4-flash:0731", "d"),
            Tier.PRO: ModelSpec("deepseek-v4-pro:0813", "d"),
            Tier.ULTRA: ModelSpec("kimi-k3", "d"),
        }
    )
    payload = _model_list_payload(models)
    ids = [m["id"] for m in payload["data"]]
    assert "Fast" in ids          # custom name advertised
    assert "mini" not in ids      # key replaced by name
    assert "air" in ids           # no name -> key used
    # internal tier key preserved
    fast = next(m for m in payload["data"] if m["id"] == "Fast")
    assert fast["tier"] == "mini"


def test_extra_params_merged_into_upstream_body(client: TestClient, monkeypatch):
    """A tier with extra_params must have those params present in the upstream body."""
    from model_router.models import RouterModels, ModelSpec, Tier

    models = RouterModels(
        tiers={
            Tier.MINI: ModelSpec("gemma4:31b", "d"),
            Tier.AIR: ModelSpec(
                "deepseek-v4-flash:0731",
                "d",
                extra_params={"reasoning_effort": "high", "budget_tokens": 4096},
            ),
            Tier.PRO: ModelSpec("deepseek-v4-pro:0813", "d"),
            Tier.ULTRA: ModelSpec("kimi-k3", "d"),
        }
    )
    settings = Settings(
        ollama_api_key="upstream-key",
        ollama_base_url="https://ollama.com/v1",
        models=models,
    )
    client = TestClient(_make_app(settings))

    seen = {}

    async def fake_post(self, url, headers, **kw):
        seen["body"] = kw["json"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    # Force the air tier (which carries extra_params) via a tier-name override.
    r = client.post(
        "/v1/chat/completions",
        json={"model": "air", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert seen["body"]["model"] == "deepseek-v4-flash:0731"
    # extra_params merged into the upstream request body
    assert seen["body"]["reasoning_effort"] == "high"
    assert seen["body"]["budget_tokens"] == 4096
