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
    assert ids == {"gemma4:31b", "deepseek-v4-flash:0731", "glm-5.2", "deepseek-v4-pro:0813"}


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
    assert r.headers["X-Router-Model"] in {"gemma4:31b", "deepseek-v4-flash:0731", "glm-5.2", "deepseek-v4-pro:0813"}
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
        "  pro-max:\n"
        "    model: my-promax\n"
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

    # /v1/models reflects the custom table.
    ids = {m["id"] for m in client.get("/v1/models").json()["data"]}
    assert ids == {"my-mini", "my-air", "my-pro", "my-promax"}

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
