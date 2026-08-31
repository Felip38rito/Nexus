#!/usr/bin/env bash
# Deprecated alias for axonctl. Kept so existing scripts/aliases keep working.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/axonctl" "$@"
