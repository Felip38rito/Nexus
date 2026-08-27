#!/usr/bin/env bash
# Deprecated alias for nexusctl. Kept so existing scripts/aliases keep working.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/nexusctl" "$@"
