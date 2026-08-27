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
PYTHONPATH=src uv run uvicorn model_router.main:app --host 127.0.0.1 --port 9000
```

Tests:

```bash
uv run pytest
```

## Environment configuration

| Var | Default | Description |
|---|---|---|
| `OLLAMA_API_KEY` | — | key for the `default` provider (Ollama Cloud) |
| `OLLAMA_BASE_URL` | `https://ollama.com/v1` | base URL for the `default` provider |
| `ROUTER_HOST` | `127.0.0.1` | router bind address |
| `ROUTER_PORT` | `9000` | router port |
| `ROUTER_DEFAULT_TIER` | `air` | last-resort fallback |
| `ROUTER_MIN_CLASSIFY_LEN` | `10` | prompts shorter than this = trivial (`mini`) |
| `ROUTER_API_KEY` | (empty) | if set, clients must send `Authorization: Bearer <key>` |
| `ROUTER_MODELS_YAML` | `router.models.yaml` | path to a custom models YAML |

> Additional providers (OpenAI, Anthropic, Gemini, …) are configured in the
> YAML `providers:` block and use their **own** env vars (e.g. `OPENAI_API_KEY`,
> `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`). `OLLAMA_API_KEY`/`OLLAMA_BASE_URL`
> only back the `default` provider.

## Model configuration (YAML)

The 4 tiers are fixed; the models, their providers, and the classifier are all
configured in `router.models.yaml` (or another file via `ROUTER_MODELS_YAML`).

```yaml
default_tier: air

# Named upstream endpoints. Each tier (and the classifier) can point at one.
# If you omit this block, a single "default" provider (Ollama Cloud) is used
# and every tier points at it.
providers:
  default:
    base_url: https://ollama.com/v1
    api_key_env: OLLAMA_API_KEY
  # gemini:
  #   base_url: https://generativelanguage.googleapis.com/v1beta
  #   api_key_env: GEMINI_API_KEY

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
    # provider: gemini        # optional — defaults to "default"
  ultra:
    model: kimi-k3
    description: "deep synthesis, whole-architecture"

classifier:
  model: gemma4:31b            # primary LLM decider (JSON decision)
  provider: default            # which provider serves the classifier
  min_classify_len: 10
```

> **IMPORTANT:** use the provider's **raw API ids** (e.g. `deepseek-v4-flash:0731`),
> not tool aliases (e.g. `deepseek-v4-flash:cloud` is a Hermes alias and returns
> 404 on the API). Check the real ids with `curl <base_url>/v1/models`.

> **Multi-provider:** each tier can point at a different provider via
> `provider: <name>`. The classifier can also run on its own provider. This
> lets you, for example, run `air` on Ollama Cloud and `pro` on Gemini/OpenAI/
> Anthropic. Each provider's key is read from its `api_key_env` variable.

## Supported providers

The router is **provider-agnostic** and **multi-provider**: you can run every
tier on one provider, or spread tiers across several. Each provider is a named
entry in the YAML `providers:` block with a `base_url` and an `api_key_env`.

To add a provider, define it in `router.models.yaml` and set its key in `.env`:

```yaml
providers:
  <name>:
    base_url: <openai-compatible /v1 endpoint>
    api_key_env: <ENV_VAR_HOLDING_THE_KEY>
```

Then point any tier at it with `provider: <name>`.

### Ollama Cloud (the default)

```yaml
providers:
  default:
    base_url: https://ollama.com/v1
    api_key_env: OLLAMA_API_KEY
tiers:
  mini:    { model: gemma4:31b,             description: "trivial/discussion" }
  air:     { model: deepseek-v4-flash:0731, description: "default day-to-day" }
  pro:     { model: deepseek-v4-pro:0813,  description: "coding power/debug" }
  ultra:   { model: kimi-k3,                description: "whole-architecture/synthesis" }
classifier:
  model: gemma4:31b
```

### Local Ollama

```yaml
providers:
  local:
    base_url: http://127.0.0.1:11434/v1
    api_key_env: OLLAMA_API_KEY   # local doesn't auth; any value
tiers:
  mini:    { model: llama3.2:3b,          description: "trivial/discussion" }
  air:     { model: qwen2.5:7b,           description: "default day-to-day" }
  pro:     { model: qwen2.5:32b,          description: "reasoning/design" }
  ultra:   { model: qwen2.5:72b,          description: "hard bug/refactor" }
classifier:
  model: llama3.2:3b
```

### OpenAI

```yaml
providers:
  openai:
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
tiers:
  mini:    { model: gpt-4o-mini,          description: "trivial/discussion" }
  air:     { model: gpt-4o-mini,          description: "default day-to-day" }
  pro:     { model: gpt-4o,               description: "reasoning/design" }
  ultra:   { model: gpt-4o,               description: "hard bug/refactor" }
classifier:
  model: gpt-4o-mini
```

### Anthropic

```yaml
providers:
  anthropic:
    base_url: https://api.anthropic.com/v1
    api_key_env: ANTHROPIC_API_KEY
tiers:
  mini:    { model: claude-3-5-haiku-latest, description: "trivial/discussion" }
  air:     { model: claude-3-5-haiku-latest, description: "default day-to-day" }
  pro:     { model: claude-3-5-sonnet-latest, description: "reasoning/design" }
  ultra:   { model: claude-3-7-sonnet-latest, description: "hard bug/refactor" }
classifier:
  model: claude-3-5-haiku-latest
```

> **Note:** Anthropic's native API is **not** OpenAI-compatible (it uses a
> different request/response shape). To route Anthropic through Nexus, use an
> OpenAI-compatible gateway in front of it (e.g. OpenRouter, or Anthropic's
> own `/v1/messages` is not supported directly). The example above assumes an
> OpenAI-compatible endpoint.

### OpenRouter

```yaml
providers:
  openrouter:
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
tiers:
  mini:    { model: meta-llama/llama-3.1-8b-instruct, description: "trivial/discussion" }
  air:     { model: anthropic/claude-3.5-haiku,       description: "default day-to-day" }
  pro:     { model: anthropic/claude-3.5-sonnet,      description: "reasoning/design" }
  ultra:   { model: anthropic/claude-3.7-sonnet,      description: "hard bug/refactor" }
classifier:
  model: meta-llama/llama-3.1-8b-instruct
```

### Groq

```yaml
providers:
  groq:
    base_url: https://api.groq.com/openai/v1
    api_key_env: GROQ_API_KEY
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

### Mixing providers across tiers

You can spread tiers across providers. Example: `air` on Ollama Cloud, `pro`
on OpenAI, `ultra` on Anthropic:

```yaml
providers:
  default:  { base_url: https://ollama.com/v1, api_key_env: OLLAMA_API_KEY }
  openai:   { base_url: https://api.openai.com/v1, api_key_env: OPENAI_API_KEY }
  anthropic:{ base_url: <openai-compatible anthropic gateway>, api_key_env: ANTHROPIC_API_KEY }
tiers:
  mini:    { model: gemma4:31b,             description: "trivial/discussion" }
  air:     { model: deepseek-v4-flash:0731, description: "default day-to-day" }
  pro:     { model: gpt-4o, provider: openai, description: "reasoning/design" }
  ultra:   { model: claude-3-7-sonnet-latest, provider: anthropic, description: "hard bug/refactor" }
classifier:
  model: gemma4:31b
```

## Using it (clients)

### curl

```bash
# List models
curl http://127.0.0.1:9000/v1/models

# Chat (streaming) — note the x-router-model / x-router-tier headers
curl http://127.0.0.1:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"adaptive","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

> `model: "adaptive"` is a virtual id that **always** forces classification.
> You can also send a tier name (`mini`, `air`, `pro`, `ultra`) to force that
> tier, or one of the raw upstream api ids (e.g. `deepseek-v4-pro:0813`) — the
> router honors it directly without re-classifying (transparent mode).

### Hermes

Add the router as a **named provider** so Hermes discovers the tiers from the
router itself (the router is the source of truth — no model list hardcoded).

#### 1. Register the provider in `config.yaml`

```bash
# Named provider entry. The `api` URL points at the router.
hermes config set providers.router.api http://127.0.0.1:9000/v1
hermes config set providers.router.name "Model Router"
hermes config set providers.router.default_model adaptive
```

> Do **NOT** set `providers.router.models`. Leaving it out lets Hermes
> *live-probe* `GET /v1/models` on the router and use exactly what the router
> advertises (`adaptive, mini, air, pro, ultra`). If you hardcode a
> comma-separated string, Hermes treats it as a single literal model id and the
> picker shows one broken row. To pin a specific list instead, set
> `providers.router.discover_models false` and provide a proper YAML list.

#### 2. Make it the active provider

```bash
hermes config set model.provider router
hermes config set model.default_model adaptive
```

#### 3. Pick a model (tiers show up)

```bash
hermes model
```

The picker now lists the router with **5 selectable models**:

```
⚙ Model Picker — Model Router
Select a model (5 available) — type to filter
> adaptive    # router decides the tier automatically
  mini        # force gemma4:31b
  air         # force deepseek-v4-flash:0731
  pro         # force deepseek-v4-pro:0813
  ultra       # force kimi-k3
```

`adaptive` routes every request through the classifier; the four tier entries
force that tier directly (transparent mode).

#### 4. Optional: short aliases

To use `/model pro` / `/model air` / etc. instead of the full ids:

```bash
hermes config set model.aliases.mini router/mini
hermes config set model.aliases.air router/air
hermes config set model.aliases.pro router/pro
hermes config set model.aliases.ultra router/ultra
```

> ⚠ Adding the router as your Hermes provider sends **all** Hermes traffic
> through it. To undo, restore the previous provider:
> `hermes config set model.provider ollama-cloud` (or whichever you used).

### Copilot CLI

The Copilot CLI speaks the **Responses API**, which the router implements
natively at `/v1/responses` — no bridge needed. Point the Copilot provider at
the router with environment variables (e.g. in `~/.zshrc`):

```bash
export COPILOT_PROVIDER_BASE_URL=http://127.0.0.1:9000/v1
export COPILOT_PROVIDER_API_KEY=router    # router without auth accepts any value
export COPILOT_PROVIDER_WIRE_API=responses  # use the /v1/responses shim
export COPILOT_MODEL=adaptive              # the router decides the tier for each request
```

- `COPILOT_PROVIDER_WIRE_API=responses` is what makes the router's
  `/v1/responses` shim handle the traffic (it translates to `/v1/chat/completions`
  upstream). Copilot also works over `chat_completions` if you prefer.
- `COPILOT_MODEL=adaptive` routes **every** request through the classifier
  (the router decides the cheapest adequate tier). To pin a tier instead, set it
  to `mini`, `air`, `pro`, or `ultra` — or to a raw upstream api id
  (e.g. `deepseek-v4-pro:0813`) for transparent mode.
- The tiers resolve to Ollama Cloud, so the router's `.env` still needs the
  upstream key (`OLLAMA_API_KEY`).
- If you enable `ROUTER_API_KEY`, set `COPILOT_PROVIDER_API_KEY` to the real
  bearer token.

> ⚠ Pointing Copilot at the router sends **all** Copilot traffic through it
> (including tool-call loops). That's the point — the classifier keeps the
> cheap tiers on the day-to-day and escalates to `pro`/`ultra` only when a
> prompt truly needs it. To go back to Copilot's cloud models, unset these
> variables (`unset COPILOT_PROVIDER_BASE_URL COPILOT_PROVIDER_API_KEY
> COPILOT_PROVIDER_WIRE_API COPILOT_MODEL`).

### Syncing the tiers into OpenCode

Unlike Hermes, **OpenCode does not auto-discover models** for custom
OpenAI-compatible providers (it requires a statically declared `models` block
per provider — see [anomalyco/opencode#27553](https://github.com/anomalyco/opencode/issues/27553)).
So the router can't be a *live* source of truth for OpenCode the way it is for
Hermes. Instead, keep the router as the single place that defines the tiers and
push a **snapshot** of its list into OpenCode with `sync-opencode.py`.

The script:
- Reads the tiers advertised by the router (`GET /v1/models`) — **the router
  stays the source of truth**.
- Rewrites **only** the `models` block of the `router` provider in your
  `opencode.json(c)`. Everything else (other providers, top-level keys, the
  `router` provider's `npm`/`name`/`options`) is preserved byte-for-byte.
- Backs up the config before writing (timestamped `.bak-<timestamp>` next to
  it), so it is **non-destructive** — your user config is never deleted.

```bash
# List the tiers the router currently advertises (requires the router running)
python3 sync-opencode.py --dry-run

# Write them into ~/.config/opencode/opencode.jsonc (default config)
python3 sync-opencode.py

# Target a different config and/or router
python3 sync-opencode.py --config /path/to/opencode.json --router-url http://127.0.0.1:9000/v1
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--config` | `~/.config/opencode/opencode.jsonc` | OpenCode config to update |
| `--router-url` | `http://127.0.0.1:9000/v1` | Nexus router base URL |
| `--dry-run` | off | Show what would change without writing |

> **How to configure the tiers (mini/air/pro/ultra):** the tier *names* come
> straight from the router's `/v1/models` (edit `router.models.yaml` — see
> [Model configuration](#model-configuration-yaml) — then restart with
> `routerctl.sh restart`). Re-run the sync and the OpenCode provider updates to
> match. If the router advertises a new tier, it appears; if one is removed,
> it disappears from the OpenCode provider too.

> **How to run it correctly:** the router must be up (`routerctl.sh status`)
> before syncing. If it's down, the script fails fast with a clear message and
> leaves your config untouched. To keep OpenCode always in step, run the sync
> after every router restart (or whenever you change `router.models.yaml`).

> **Non-destructive guarantee:** the script only ever replaces the `models`
> value inside the `router` provider. It never deletes other providers or other
> keys in your config, and it always leaves a backup before the first write.

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
        "mini": { "name": "Gemma 4 31B (mini)", "limit": { "context": 1048576, "output": 65536 } },
        "air": { "name": "DeepSeek V4 Flash (air)", "limit": { "context": 1048576, "output": 65536 } },
        "pro": { "name": "DeepSeek V4 Pro (pro)", "limit": { "context": 1048576, "output": 65536 } },
        "ultra": { "name": "Kimi K3 (ultra)", "limit": { "context": 1048576, "output": 65536 } }
      }
    }
  }
}
```

> The router doesn't require auth by default, so `apiKey` can be any non-empty
> value. If you enable `ROUTER_API_KEY`, replace it with the real value.
>
> **Tip:** don't hand-edit this block — run `sync-opencode.py` (see
> [Syncing the tiers into OpenCode](#syncing-the-tiers-into-opencode)) so the
> model list always matches what the router advertises.

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

- Provider keys are read from env (or `.env`), **never** committed.
- `.gitignore` covers `.env`, `.venv/`, `logs/`, `router.log`.
- `yaml.safe_load` (no YAML RCE).
- Default bind on `127.0.0.1` (not exposed to the network).
- Optional auth via `ROUTER_API_KEY` (Bearer).
- Each provider's key lives in its own env var (e.g. `OLLAMA_API_KEY`,
  `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) — set them in `.env`, which is git-ignored.

## Tests / verification

- 49 unit tests (`classify`, `config`, `proxy`, `sync-opencode`) with MockTransport.
- Real smoke test against Ollama Cloud validated 2026-08-25.
