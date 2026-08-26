#!/usr/bin/env bash
# Management script for the Model Router launchd agent.

set -euo pipefail

PLIST="$HOME/Library/LaunchAgents/br.com.felp38rito.nexus.plist"
SERVICE="br.com.felp38rito.nexus"
LOG_OUT="$HOME/Developer/model-router/logs/router.out.log"
LOG_ERR="$HOME/Developer/model-router/logs/router.err.log"
DOMAIN="gui/$(id -u)"

usage() {
    echo "Usage: routerctl [start|stop|restart|status|logs|tail]"
    exit 1
}

is_loaded() {
    launchctl list "$SERVICE" &>/dev/null
}

case "${1:-}" in
    start)
        if is_loaded; then
            echo "ℹ️  Model Router already running."
        else
            echo "🚀 Starting Model Router..."
            launchctl bootstrap "$DOMAIN" "$PLIST"
            echo "Done. Check status with 'routerctl status'."
        fi
        ;;
    stop)
        if is_loaded; then
            echo "🛑 Stopping Model Router..."
            launchctl bootout "$DOMAIN/$SERVICE"
            # bootout is async — wait until the service is actually gone.
            for _ in $(seq 1 20); do
                is_loaded || break
                sleep 0.25
            done
            echo "Done."
        else
            echo "ℹ️  Model Router not running."
        fi
        ;;
    restart)
        "$0" stop
        "$0" start
        ;;
    status)
        if is_loaded; then
            echo "🔍 Model Router is running:"
            launchctl list "$SERVICE"
        else
            echo "🔍 Model Router is NOT running."
        fi
        ;;
    logs)
        echo "📄 Full logs:"
        tail -n 100 "$LOG_OUT"
        echo "--- Errors ---"
        tail -n 100 "$LOG_ERR"
        ;;
    tail)
        tail -f "$LOG_OUT" "$LOG_ERR"
        ;;
    *)
        usage
        ;;
esac
