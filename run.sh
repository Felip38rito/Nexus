#!/usr/bin/env bash
# Sobe o router lendo a OLLAMA_API_KEY de ~/.hermes/.env (fora do repo).
# O .env do Hermes não entra no git, então nenhuma credencial vaza pro repo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Carrega OLLAMA_API_KEY de ~/.hermes/.env (é a fonte da key do usuário).
HERMES_ENV="${HERMES_ENV:-$HOME/.hermes/.env}"
if [ -f "$HERMES_ENV" ]; then
  # shellcheck disable=SC1090
  set -a; source "$HERMES_ENV"; set +a
fi

export PYTHONPATH="${PYTHONPATH:-}:$SCRIPT_DIR/src"
exec uv run uvicorn model_router.main:app \
  --host "${ROUTER_HOST:-127.0.0.1}" \
  --port "${ROUTER_PORT:-9000}"
