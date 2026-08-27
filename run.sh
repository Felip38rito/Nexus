#!/usr/bin/env bash
# Sobe o router. Lê a chave do provider de .env (no repo) — ou, como fallback
# legado, de ~/.hermes/.env. Nenhuma credencial vai pro git.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1) Repo-local .env (padrão para instalações novas / "a galera").
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a; source "$SCRIPT_DIR/.env"; set +a
# 2) Fallback legado: .env do Hermes (setup original do Felipe).
elif [ -f "${HERMES_ENV:-$HOME/.hermes/.env}" ]; then
  set -a; source "${HERMES_ENV:-$HOME/.hermes/.env}"; set +a
fi

export PYTHONPATH="${PYTHONPATH:-}:$SCRIPT_DIR/src"
exec uv run uvicorn model_router.main:app \
  --host "${ROUTER_HOST:-127.0.0.1}" \
  --port "${ROUTER_PORT:-9000}"
