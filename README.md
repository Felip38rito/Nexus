# Nexus

A local **OpenAI-compatible** proxy that routes each chat request to the
cheapest model that can handle the task. It keeps the expensive models
(`pro`/`ultra`) reserved for prompts that truly need them, and sends
trivial/day-to-day requests to the cheap ones.

It is **provider-agnostic**: it works with any OpenAI-compatible API (Ollama
Cloud, local Ollama, OpenAI, OpenRouter, Groq, Together, …). You just set the
`base_url` + key of your provider and the tier → model mapping.

## Why Nexus?

The problem Nexus solves is simple and expensive: **paying for a Pro model to
answer "hello"**.

Without a router, you have two bad choices:

- **Pin to the cheap model** → complex prompts (architecture refactor, race
  condition debugging) get weak answers and you waste time.
- **Pin to the expensive model** → every request, even trivial ones, pays the
  top-tier price. In an agent (Hermes, OpenCode, Cursor) that makes dozens of
  tool calls per task, that burns credits for nothing.

Nexus breaks that trade-off: it **classifies the intent** of each request and
routes to the cheapest tier that can handle it. Trivial goes to `mini`,
day-to-day to `air`, and heavy reasoning only then climbs to `pro`/`ultra`.

The result is **expensive-model performance at cheap-model cost** — the same
answer quality, without wasting tokens on prompts that don't need it.

## How it works

1. A client (Hermes, OpenCode, curl, any OpenAI SDK) points its `base_url` at the router.
2. The router reads the prompt and classifies the difficulty (**hybrid** classifier):
   - **Deterministic** only for the obvious: explicit model override ("use
     deepseek-v4-pro") and trivial chatter (greetings/short). Instant, free.
   - **LLM-primary** for everything else: a cheap model returns strict JSON
     with the qualitative tier decision. Tiering is fundamentally qualitative
     (intent, scope, context) — keyword matching can't capture that.
   - **Fail-safe**: LLM error → default `air`. Never breaks the request.
3. The router forwards to the chosen model and streams the response.
   - Headers: `X-Router-Model` (model id) and `X-Router-Tier` (mini/air/pro/ultra).

> **Only the last user message** feeds the classifier. The system prompt and
> tool-call history are ignored in the tier decision — otherwise accumulated
> technical context would saturate everything to `pro`/`ultra`.

## Tiers

| Tier | Use |
|---|---|
| `mini` | trivial/mechanical + discussion |
| `air` (default) | day-to-day |
| `pro` | complex reasoning / coding power / hard debug / refactor / concurrency / public API |
| `ultra` | hardest problems, whole-architecture, deep synthesis, adversarial |

## Requirements

- Python ≥ 3.11 via [`uv`](https://docs.astral.sh/uv/).
- An upstream provider API key (e.g. `OLLAMA_API_KEY`).

## Quick start

```bash
git clone <your-repo> && cd model-router
uv sync --extra dev
cp .env.example .env        # fill in your provider key
PYTHONPATH=src uv run uvicorn model_router.main:app --host 127.0.0.1 --port 8000
```

Tests:

```bash
uv run pytest
```

## Environment configuration

| Var | Default | Description |
|---|---|---|
| `OLLAMA_API_KEY` | — | upstream provider key (required) |
| `OLLAMA_BASE_URL` | `https://ollama.com/v1` | provider `/v1` endpoint |
| `ROUTER_HOST` | `127.0.0.1` | router bind address |
| `ROUTER_PORT` | `8000` | router port |
| `ROUTER_DEFAULT_TIER` | `air` | last-resort fallback |
| `ROUTER_MIN_CLASSIFY_LEN` | `10` | prompts shorter than this = trivial (`mini`) |
| `ROUTER_API_KEY` | (empty) | if set, clients must send `Authorization: Bearer <key>` |
| `ROUTER_MODELS_YAML` | `router.models.yaml` | path to a custom models YAML |

## Model configuration (YAML)

The 4 tiers are fixed; only the models/descriptions change. Edit
`router.models.yaml` (or point to another with `ROUTER_MODELS_YAML`):

```yaml
default_tier: air
tiers:
  mini:
    model: gemma4:31b          # provider's real API id
    description: "fast/cheap - discussion + trivial/mechanical"
  air:
    model: deepseek-v4-flash:0731
    description: "default - day-to-day"
  pro:
    model: deepseek-v4-pro:0813
    description: "raw coding power"
  ultra:
    model: kimi-k3
    description: "deep synthesis, whole-architecture"
classifier:
  model: gemma4:31b            # primary LLM decider (JSON decision)
  min_classify_len: 10
```

> **IMPORTANT:** use the provider's **raw API ids** (e.g. `deepseek-v4-flash:0731`),
> not tool aliases (e.g. `deepseek-v4-flash:cloud` is a Hermes alias and returns
> 404 on the API). Check the real ids with `curl <base_url>/v1/models`.

> **The classifier also runs on the provider.** `classifier.model` must be a
> model the provider has (preferably a cheap one). It uses the same
> `OLLAMA_BASE_URL`/`OLLAMA_API_KEY` as the upstream.

## Supported providers

The router is agnostic. To switch providers, change **3 things**:
`OLLAMA_BASE_URL`, `OLLAMA_API_KEY`, and the tier mapping in the YAML.

### Ollama Cloud (the example used here)

```bash
export OLLAMA_API_KEY="<your-key>"
export OLLAMA_BASE_URL="https://ollama.com/v1"
```

```yaml
tiers:
  mini:    { model: gemma4:31b,             description: "trivial/discussion" }
  air:     { model: deepseek-v4-flash:0731, description: "default day-to-day" }
  pro:     { model: deepseek-v4-pro:0813,  description: "coding power/debug" }
  ultra:   { model: kimi-k3,                description: "whole-architecture/synthesis" }
classifier:
  model: gemma4:31b
```

### Local Ollama

```bash
export OLLAMA_API_KEY="ollama"            # local doesn't auth; any value
export OLLAMA_BASE_URL="http://127.0.0.1:11434/v1"
```

```yaml
tiers:
  mini:    { model: llama3.2:3b,          description: "trivial/discussion" }
  air:     { model: qwen2.5:7b,           description: "default day-to-day" }
  pro:     { model: qwen2.5:32b,          description: "reasoning/design" }
  ultra:   { model: qwen2.5:72b,          description: "hard bug/refactor" }
classifier:
  model: llama3.2:3b
```

### OpenAI

```bash
export OLLAMA_API_KEY="sk-..."            # your OpenAI key
export OLLAMA_BASE_URL="https://api.openai.com/v1"
```

```yaml
tiers:
  mini:    { model: gpt-4o-mini,          description: "trivial/discussion" }
  air:     { model: gpt-4o-mini,          description: "default day-to-day" }
  pro:     { model: gpt-4o,               description: "reasoning/design" }
  ultra:   { model: gpt-4o,               description: "hard bug/refactor" }
classifier:
  model: gpt-4o-mini
```

### OpenRouter

```bash
export OLLAMA_API_KEY="sk-or-..."         # your OpenRouter key
export OLLAMA_BASE_URL="https://openrouter.ai/api/v1"
```

```yaml
tiers:
  mini:    { model: meta-llama/llama-3.1-8b-instruct, description: "trivial/discussion" }
  air:     { model: anthropic/claude-3.5-haiku,       description: "default day-to-day" }
  pro:     { model: anthropic/claude-3.5-sonnet,      description: "reasoning/design" }
  ultra:   { model: anthropic/claude-3.7-sonnet,      description: "hard bug/refactor" }
classifier:
  model: meta-llama/llama-3.1-8b-instruct
```

### Groq

```bash
export OLLAMA_API_KEY="gsk_..."           # your Groq key
export OLLAMA_BASE_URL="https://api.groq.com/openai/v1"
```

```yaml
tiers:
  mini:    { model: llama-3.1-8b-instant, description: "trivial/discussion" }
  air:     { model: llama-3.3-70b-versatile, description: "default day-to-day" }
  pro:     { model: llama-3.3-70b-versatile, description: "reasoning/design" }
  ultra:   { model: llama-3.3-70b-versatile, description: "hard bug/refactor" }
classifier:
  model: llama-3.1-8b-instant
```

> The model ids above are examples — check your provider's real ids with
> `curl <base_url>/v1/models` before using them.

## Using it (clients)

### curl

```bash
# List models
curl http://127.0.0.1:8000/v1/models

# Chat (streaming) — note the x-router-model / x-router-tier headers
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"adaptive","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

> `model: "adaptive"` is a virtual id that **always** forces classification.
> If you send one of the real tier ids (e.g. `deepseek-v4-pro:0813`), the router
> honors it directly without re-classifying (transparent mode).

### Hermes

```bash
hermes config set model.provider openai-compatible
hermes config set model.base_url http://127.0.0.1:8000
```

⚠ Adds a hop to ALL Hermes traffic. Undo with
`hermes config set model.base_url https://ollama.com/v1` (or your original value).

### OpenCode

Add a `router` provider in `~/.config/opencode/opencode.jsonc`:

```jsonc
{
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
        "gemma4:31b": { "name": "Gemma 4 31B (mini)", "limit": { "context": 1048576, "output": 65536 } },
        "deepseek-v4-flash:0731": { "name": "DeepSeek V4 Flash (air)", "limit": { "context": 1048576, "output": 65536 } },
        "deepseek-v4-pro:0813": { "name": "DeepSeek V4 Pro (pro)", "limit": { "context": 1048576, "output": 65536 } },
        "kimi-k3": { "name": "Kimi K3 (ultra)", "limit": { "context": 1048576, "output": 65536 } }
      }
    }
  }
}
```

> The router doesn't require auth by default, so `apiKey` can be any non-empty
> value. If you enable `ROUTER_API_KEY`, replace it with the real value.

## Running as a service (launchd — macOS)

To have the router start at login and restart itself on crash, use a LaunchAgent:

- Plist: `~/Library/LaunchAgents/br.com.felp38rito.nexus.plist`
  (`RunAtLoad` + `KeepAlive` + `ThrottleInterval=10`; logs in `logs/`).
- Wrapper: `routerctl.sh` — `start | stop | restart | status | logs | tail`.

```bash
./routerctl.sh status     # is it running?
./routerctl.sh restart    # after changing code/config
./routerctl.sh tail       # follow logs live
```

> **Pitfall:** launchd doesn't source your `~/.zshrc`, so its minimal PATH
> can't find `uv` (which usually lives in `~/.local/bin`). The plist MUST set
> `EnvironmentVariables.PATH` to include the `uv` directory, otherwise the
> service crash-loops with `exec: uv: not found`.

## Security

- The provider key is read from env (or `.env`), **never** committed.
- `.gitignore` covers `.env`, `.venv/`, `logs/`, `router.log`.
- `yaml.safe_load` (no YAML RCE).
- Default bind on `127.0.0.1` (not exposed to the network).
- Optional auth via `ROUTER_API_KEY` (Bearer).

## Tests / verification

- 33 unit tests (`classify`, `config`, `proxy`) with MockTransport.
- Real smoke test against Ollama Cloud validated 2026-08-25.
