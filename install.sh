#!/usr/bin/env bash
# install.sh - Bootstrap installer for Axon model router
# Supports remote execution: curl -LsSf https://.../install.sh | sh

set -euo pipefail

# --- Configuration ---
# Overridable via env for CI/testing: AXON_REPO_URL, AXON_STABLE_HOME.
STABLE_HOME="${AXON_STABLE_HOME:-$HOME/.axon}"
REPO_URL="${AXON_REPO_URL:-https://github.com/Felip38rito/Axon}"
DEFAULT_PORT=9000
PORT="$DEFAULT_PORT"
NO_SERVICE=false

# Colors
log()  { echo -e "\033[1;32m[INFO]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error(){ echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port|-p)
                if [[ -z "${2:-}" ]]; then
                    error "--port requires a value"
                    exit 1
                fi
                PORT="$2"
                shift 2
                ;;
            --no-service)
                NO_SERVICE=true
                shift
                ;;
            -h|--help)
                echo "Usage: curl -LsSf https://.../install.sh | sh -s -- [--port N] [--no-service]"
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
}

# Interactive prompt for port if not provided and terminal is interactive
prompt_port() {
    if [[ "$PORT" == "$DEFAULT_PORT" ]] && [[ -t 0 ]]; then
        echo -n -e "\033[1;32m[INFO]\033[0m Router port [9000]: "
        read -r user_port
        if [[ -n "$user_port" ]]; then
            if [[ "$user_port" =~ ^[0-9]+$ ]] && (( user_port > 0 && user_port < 65536 )); then
                PORT="$user_port"
            else
                warn "Invalid port '$user_port'. Using default $DEFAULT_PORT."
            fi
        fi
    fi
}

# Ensure uv is available
ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        log "uv already installed: $(uv --version)"
        return
    fi
    warn "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:$PATH"
}

# Clone or update the repo in the stable home
setup_repo() {
    if [ -d "$STABLE_HOME/.git" ]; then
        log "Axon already exists in $STABLE_HOME. Updating..."
        (cd "$STABLE_HOME" && git pull)
    else
        log "Cloning Axon to $STABLE_HOME..."
        git clone "$REPO_URL" "$STABLE_HOME"
    fi
}

# Install the CLI tool globally via uv
install_cli() {
    log "Installing axon CLI tool..."
    uv tool install "$STABLE_HOME"
}

# Install the background service
install_service() {
    if $NO_SERVICE; then
        log "Skipping service installation (--no-service)."
        return
    fi
    log "Installing background service on port $PORT..."
    # Use the axon CLI we just installed to trigger the service setup
    axon install --port "$PORT"
}

main() {
    parse_args "$@"
    prompt_port
    log "Starting Axon bootstrap installation..."
    
    ensure_uv
    setup_repo
    install_cli
    install_service
    
    echo
    echo "=================================================="
    echo " Axon installation complete"
    echo "=================================================="
    echo " Stable Home: $STABLE_HOME"
    echo " Port:        $PORT"
    echo " CLI:         axon (available in your PATH)"
    echo " Service:    $([ $NO_SERVICE = true ] && echo 'not installed (--no-service)' || echo 'installed and running')"
    echo
    echo " Usage:"
    echo "   axon status    - Check service status"
    echo "   axon restart   - Restart service"
    echo "   axon logs      - View logs"
    echo "   axon setup     - Interactive configuration"
    echo "=================================================="
}

main "$@"
