#!/usr/bin/env bash
# install.sh - Bootstrap installer for Axon model router
# Works even without uv pre-installed. Installs uv, syncs deps, sets up .env,
# adds repo to PATH, installs background service, and validates.

set -euo pipefail

# --- Configuration ---
# Resolve the real script path so REPO_DIR works even when invoked via PATH.
SCRIPT_SRC="${BASH_SOURCE[0]}"
if [[ "$SCRIPT_SRC" != /* ]]; then
    SCRIPT_SRC="$(command -v "$SCRIPT_SRC" 2>/dev/null || echo "$SCRIPT_SRC")"
fi
REPO_DIR="$(cd "$(dirname "$SCRIPT_SRC")" && pwd)"
DEFAULT_PORT=9000
PORT="$DEFAULT_PORT"
NO_SERVICE=false
DRY_RUN=false

# --- Helper functions ---
log()  { echo -e "\033[1;32m[INFO]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error(){ echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

# Parse command-line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --no-service) NO_SERVICE=true ;;
            --port)
                if [[ -z "${2:-}" ]]; then
                    error "--port requires a value"
                    exit 1
                fi
                if ! [[ "$2" =~ ^[0-9]+$ ]] || (( $2 < 1 || $2 > 65535 )); then
                    error "--port must be an integer between 1 and 65535 (got: $2)"
                    exit 1
                fi
                PORT="$2"
                shift
                ;;
            --dry-run) DRY_RUN=true ;;
            -h|--help)
                echo "Usage: $0 [--no-service] [--port N] [--dry-run]"
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                exit 1
                ;;
        esac
        shift
    done
}

# Ensure uv is available; install if missing
ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        log "uv already installed: $(uv --version)"
        return
    fi

    warn "uv not found. Attempting to install via Astral installer..."
    if ! command -v curl >/dev/null 2>&1; then
        error "curl is required to install uv. Please install curl and rerun."
        exit 1
    fi

    if $DRY_RUN; then
        log "[DRY RUN] Would run: curl -LsSf https://astral.sh/uv/install.sh | sh"
        return
    fi

    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Refresh PATH: uv installs to ~/.local/bin on macOS/Linux
    export PATH="${HOME:-}/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        error "uv installation failed. Please install manually."
        exit 1
    fi
    log "uv installed successfully: $(uv --version)"
}

# Sync dependencies
sync_deps() {
    log "Syncing dependencies with uv..."
    if $DRY_RUN; then
        log "[DRY RUN] Would run: uv sync --extra dev"
        return
    fi
    (cd "$REPO_DIR" && uv sync --extra dev)
    log "Dependencies synced."
}

# Create .env from .env.example if missing
setup_env() {
    local env_file="$REPO_DIR/.env"
    local example_file="$REPO_DIR/.env.example"
    if [[ -f "$env_file" ]]; then
        warn ".env already exists. Skipping creation (non-destructive)."
    else
        if [[ -f "$example_file" ]]; then
            log "Creating .env from .env.example"
            if $DRY_RUN; then
                log "[DRY RUN] Would copy $example_file to $env_file"
            else
                cp "$example_file" "$env_file"
            fi
        else
            warn ".env.example not found. Skipping .env creation."
        fi
    fi
}

# Add repo dir to shell PATH if not already present
add_to_path() {
    local shell_config=""
    local shell_name="$(basename "${SHELL:-}")"
    case "$shell_name" in
        zsh) shell_config="$HOME/.zshrc" ;;
        bash) shell_config="$HOME/.bashrc" ;;
        fish) shell_config="$HOME/.config/fish/config.fish" ;;
        *) warn "Unsupported shell '$shell_name'. Skipping PATH addition."; return ;;
    esac

    if [[ -f "$shell_config" ]]; then
        if grep -qF "export PATH=\"$REPO_DIR:\$PATH\"" "$shell_config" \
           || grep -qF "fish_add_path \"$REPO_DIR\"" "$shell_config"; then
            log "Repo directory already in PATH (via $shell_config)."
        else
            log "Adding repo directory to PATH in $shell_config"
            if $DRY_RUN; then
                log "[DRY RUN] Would append PATH line to $shell_config"
            else
                # Ensure a trailing newline so we never corrupt the last line.
                [[ -s "$shell_config" ]] && [[ "$(tail -c1 "$shell_config")" != "" ]] \
                    && printf '\n' >> "$shell_config"
                if [[ "$shell_name" == "fish" ]]; then
                    echo "fish_add_path \"$REPO_DIR\"" >> "$shell_config"
                else
                    echo "export PATH=\"$REPO_DIR:\$PATH\"" >> "$shell_config"
                fi
            fi
        fi
    else
        warn "Shell config file $shell_config not found. Skipping PATH addition."
    fi
}

# Install background service via axonctl
install_service() {
    if $NO_SERVICE; then
        log "Skipping service installation (--no-service)."
        return
    fi
    log "Installing background service via axonctl..."
    if $DRY_RUN; then
        log "[DRY RUN] Would run: $REPO_DIR/axonctl install --port $PORT"
        return
    fi
    (cd "$REPO_DIR" && ./axonctl install --port "$PORT")
    log "Service installed."
}

# Validate the /v1/models endpoint
validate() {
    if $NO_SERVICE; then
        log "Skipping validation (--no-service)."
        return
    fi
    local url="http://localhost:$PORT/v1/models"
    log "Validating endpoint: $url"
    if $DRY_RUN; then
        log "[DRY RUN] Would curl $url"
        return
    fi
    if command -v curl >/dev/null 2>&1; then
        if curl -fsS "$url" >/dev/null 2>&1; then
            log "Validation successful: endpoint is reachable."
        else
            warn "Validation failed: endpoint not reachable. Check service status."
        fi
    else
        warn "curl not found. Skipping validation."
    fi
}

# Print final summary
print_summary() {
    echo
    echo "=================================================="
    if $DRY_RUN; then
        echo " Axon installation (DRY RUN — nothing was changed)"
    else
        echo " Axon installation complete"
    fi
    echo "=================================================="
    echo " Repository: $REPO_DIR"
    echo " Port: $PORT"
    echo " Service: $([ $NO_SERVICE = true ] && echo 'not installed' || echo 'installed')"
    echo
    echo " Usage:"
    echo "   axonctl status   - Check service status"
    echo "   axonctl restart  - Restart service"
    echo "   axonctl logs     - View logs"
    echo "   axonctl tail     - Follow logs"
    echo
    echo " To uninstall:"
    echo "   axonctl uninstall"
    echo "   (and remove the PATH line from your shell config)"
    echo "=================================================="
}

# --- Main ---
main() {
    parse_args "$@"
    cd "$REPO_DIR"

    log "Starting Axon installation..."
    ensure_uv
    sync_deps
    setup_env
    add_to_path
    install_service
    validate
    print_summary
}

main "$@"
