# Model Router

Proxy local **OpenAI-compatível** que roteia cada chat request pro modelo
Ollama Cloud mais barato que dá conta da tarefa. Deixa os modelos caros
(`pro`/`pro-max`) reservados pros prompts que realmente exigem, e manda
trivial/day-to-day pros baratos.

Ferramenta **global do usuário** — não pertence a nenhum projeto específico.
Mora em `~/Developer/model-router/`, fora do repo Aura.

## Como funciona

1. Cliente (Hermes, OpenCode, curl, qualquer OpenAI SDK) aponta `base_url` pro router.
2. Router lê o prompt e classifica a dificuldade (classificador **híbrido**):
   - **Determinístico** só para óbvios: override explícito de modelo ("use
     deepseek-v4-pro") e chatter trivial (saudações/curtos). Instantâneo, grátis.
   - **LLM primário** para todo o resto: `gemma4:31b` retorna JSON estrito com a
     decisão qualitativa de tier. A decisão de tier é fundamentalmente
     qualitativa (intenção, escopo, contexto) — keyword matching não captura isso.
   - **Fail-safe**: erro do LLM → default `air`. Nunca quebra o request.
3. Router repassa pro modelo escolhido e faz streaming da resposta.
   - Headers: `X-Router-Model` (id do modelo) e `X-Router-Tier` (mini/air/pro/pro-max).

## Tiers → modelos (API ids reais do `/v1/models`)

| Tier | Modelo | Uso |
|---|---|---|
| mini | `gemma4:31b` | trivial/mecânico + discussão |
| air (default) | `deepseek-v4-flash:0731` | day-to-day |
| pro | `glm-5.2` | reasoning complexo / design ambíguo / concurrency / public API |
| pro-max | `deepseek-v4-pro:0813` | bug difícil, refactor pesado, arquitetura |

> **IMPORTANTE:** o sufixo `:cloud` (ex. `deepseek-v4-flash:cloud`) é **alias
> do Hermes** e NÃO existe na API — retorna 404. O router usa os ids crus.

## Requisitos

- Python ≥ 3.11 via [`uv`](https://docs.astral.sh/uv/).
- `OLLAMA_API_KEY` (upstream key da Ollama Cloud).

## Setup e execução

```bash
cd ~/Developer/model-router
uv sync --extra dev
cp .env.example .env        # preenche OLLAMA_API_KEY
PYTHONPATH=src uv run uvicorn model_router.main:app --host 127.0.0.1 --port 8000
```

Testes:

```bash
cd ~/Developer/model-router
uv run pytest
```

## Uso (curl)

```bash
# Listar modelos
curl http://127.0.0.1:8000/v1/models

# Chat (streaming)
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"whatever","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

Repare no header `x-router-model`/`x-router-tier` na resposta.

## Configuração via env

| Var | Default | Descrição |
|---|---|---|
| `OLLAMA_API_KEY` | — | key da Ollama Cloud (obrigatória) |
| `OLLAMA_BASE_URL` | `https://ollama.com/v1` | upstream |
| `ROUTER_HOST` | `127.0.0.1` | bind do router |
| `ROUTER_PORT` | `8000` | porta do router |
| `ROUTER_DEFAULT_TIER` | `air` | fallback de último recurso |
| `ROUTER_MIN_CLASSIFY_LEN` | `10` | prompts menores que isso = triviais (`mini`) |
| `ROUTER_API_KEY` | (vazio) | se setar, cliente precisa mandar `Authorization: Bearer <chave>` |
| `ROUTER_MODELS_YAML` | `router.models.yaml` | caminho pra um YAML de modelos customizado |

## Configuração de modelos (YAML)

Os modelos são configuráveis via arquivo YAML (por padrão `router.models.yaml`
na raiz do projeto, ou aponte outro com `ROUTER_MODELS_YAML`). Estrutura:

```yaml
default_tier: air
tiers:
  mini:
    model: gemma4:31b          # API id real (sem sufixo :cloud)
    description: "fast/cheap - discussion + trivial/mechanical"
  air:
    model: deepseek-v4-flash:0731
    description: "default - day-to-day"
  pro:
    model: glm-5.2
    description: "frontier reasoning"
  pro-max:
    model: deepseek-v4-pro:0813
    description: "raw coding power"
classifier:
  model: gemma4:31b             # LLM decisor primário (JSON de decisão)
  min_classify_len: 10          # prompts menores que isso = triviais (mini)
```

Os 4 tiers (mini/air/pro/pro-max) são fixos — só os modelos/descrições mudam.
O arquivo default tem comentários úteis.

## Pontar o Hermes pro router (opcional, reversível)

```bash
hermes config set model.provider openai-compatible
hermes config set model.base_url http://127.0.0.1:8000
```

⚠ Adiciona um hop em TODO o tráfego do Hermes. Desfaça com
`hermes config set model.base_url https://ollama.com/v1` (ou o valor original).
Opcional e não muda o config do usuário por padrão.

## Testes / verificação

- 32 testes unitários (`classify`, `config`, `proxy`) com MockTransport.
- Smoke real contra a Ollama Cloud validado 2026-08-25 (ver `Model Router.md` no vault).

## Próximos passos / ideias

- Levar pro GitHub (repo próprio).
- Serviço de longo prazo via `launchd`/`systemd`/Docker.
- Refinar keyword heuristics (feedback de rotas erradas).
