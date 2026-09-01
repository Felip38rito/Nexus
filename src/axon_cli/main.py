"""axon CLI — manage the Axon Model Router.

Wraps `axonctl.sh` for service lifecycle and provides config management.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

from . import __version__
from .config_cmd import list_config, set_tier_model
from .setup import run_setup

app = typer.Typer(
    name="axon",
    help="Manage the Axon Model Router: service lifecycle, setup, and config.",
    no_args_is_help=True,
)
console = Console()

# Repo root: src/axon_cli/main.py -> src -> repo root.
# Prefer stable home (~/.axon), fallback to local repo structure.
STABLE_HOME = Path.home() / ".axon"
REPO_DIR = STABLE_HOME if STABLE_HOME.exists() else Path(__file__).resolve().parents[2]
AXONCTL = REPO_DIR / "axonctl.sh"


def _find_axonctl() -> Path:
    """Locate axonctl.sh: prefer the repo copy, else a PATH-installed one."""
    if AXONCTL.exists():
        return AXONCTL
    found = shutil.which("axonctl")
    if found:
        return Path(found)
    console.print(
        "[red]axonctl.sh not found.[/red] "
        f"Expected at {AXONCTL} or on PATH.",
        file=sys.stderr,
    )
    raise typer.Exit(code=1)


def _run_axonctl(*args: str) -> None:
    """Proxy a command to axonctl.sh, streaming its output."""
    script = _find_axonctl()
    try:
        proc = subprocess.run(
            [str(script), *args],
            cwd=REPO_DIR,
        )
    except FileNotFoundError:
        console.print(f"[red]Could not execute {script}[/red]", file=sys.stderr)
        raise typer.Exit(code=1)
    if proc.returncode != 0:
        raise typer.Exit(code=proc.returncode)


@app.command()
def version() -> None:
    """Print the CLI version."""
    console.print(f"axon {__version__}")


@app.command()
def setup() -> None:
    """Run the interactive setup to configure the router."""
    run_setup()


@app.command()
def config(
    action: str = typer.Argument(..., help="'list' or 'set'"),
    tier: str = typer.Argument(None, help="Tier key (for 'set')"),
    model: str = typer.Option(None, "--model", "-m", help="Model id (for 'set')"),
) -> None:
    """View or modify the router config."""
    if action == "list":
        list_config()
    elif action == "set":
        if not tier or not model:
            console.print(
                "[red]Usage: axon config set <tier> --model <id>[/red]",
                file=sys.stderr,
            )
            raise typer.Exit(code=1)
        set_tier_model(tier, model)
    else:
        console.print(
            f"[red]Unknown config action '{action}'.[/red] Use 'list' or 'set'.",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)


@app.command()
def start() -> None:
    """Start the Axon router service."""
    _run_axonctl("start")


@app.command()
def install(
    port: int = typer.Option(9000, "--port", "-p", help="Port to run the router on"),
) -> None:
    """Generate the service config and load it (start at login)."""
    _run_axonctl("install", "--port", str(port))


@app.command()
def uninstall() -> None:
    """Stop and remove the service config."""
    _run_axonctl("uninstall")


@app.command()
def stop() -> None:
    """Stop the Axon router service."""
    _run_axonctl("stop")


@app.command()
def restart() -> None:
    """Restart the Axon router service."""
    _run_axonctl("restart")


@app.command()
def status() -> None:
    """Show whether the Axon router service is running."""
    _run_axonctl("status")


@app.command()
def logs() -> None:
    """Print the last 100 lines of router logs."""
    _run_axonctl("logs")


@app.command()
def tail() -> None:
    """Follow the router logs live."""
    _run_axonctl("tail")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
