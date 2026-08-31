"""Interactive setup flow for the Axon Model Router.

Guides the user through configuring providers, tiers, and reasoning levels,
then writes the resulting config to ~/.config/axon/config.yml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from model_router.models import Tier

console = Console()

# The four tier roles, explained in plain English.
TIER_ROLES: dict[Tier, str] = {
    Tier.MINI: "trivial/mechanical tasks — quick, cheap, no deep thought",
    Tier.AIR: "standard implementation — the day-to-day default",
    Tier.PRO: "high-cognitive / unclear path — hard debugging, refactors",
    Tier.ULTRA: "systemic synthesis / hardest problems — whole-architecture",
}

DEFAULT_MODELS: dict[Tier, str] = {
    Tier.MINI: "gemma4:31b",
    Tier.AIR: "deepseek-v4-flash:0731",
    Tier.PRO: "deepseek-v4-pro:0813",
    Tier.ULTRA: "kimi-k3",
}

DEFAULT_PROVIDER = "default"
DEFAULT_BASE_URL = "https://ollama.com/v1"
DEFAULT_API_KEY_ENV = "OLLAMA_API_KEY"


def config_path() -> Path:
    """The user config file location: ~/.config/axon/config.yml."""
    return Path.home() / ".config" / "axon" / "config.yml"


def _print_roles() -> None:
    """Display a brief explanation of the 4 tier roles."""
    lines = []
    for tier in Tier:
        lines.append(f"[bold]{tier.value}[/bold] — {TIER_ROLES[tier]}")
    console.print(
        Panel(
            "\n".join(lines),
            title="Axon Tiers",
            subtitle="Each request is routed to the cheapest adequate tier",
        )
    )


def _prompt_provider() -> dict[str, Any]:
    """Collect provider configuration from the user."""
    console.print(Panel("Provider configuration", border_style="cyan"))
    name = Prompt.ask("Provider name", default="default")
    base_url = Prompt.ask("base_url", default=DEFAULT_BASE_URL)
    api_key_env = Prompt.ask(
        "Environment variable holding the API key",
        default=DEFAULT_API_KEY_ENV,
    )
    return {
        "name": name,
        "base_url": base_url.rstrip("/"),
        "api_key_env": api_key_env,
    }


def _prompt_tiers() -> dict[Tier, dict[str, Any]]:
    """Collect tier names and model ids (Omakase vs Custom flow)."""
    _print_roles()
    omakase = Confirm.ask(
        "Use Axon Omakase (recommended defaults) or Custom Names?",
        default=True,
    )

    tiers: dict[Tier, dict[str, Any]] = {}
    for tier in Tier:
        entry: dict[str, Any] = {}
        if omakase:
            # Omakase: use the tier key as the display name.
            entry["name"] = tier.value
        else:
            entry["name"] = Prompt.ask(
                f"Display name for tier [bold]{tier.value}[/bold] ({TIER_ROLES[tier]})",
                default=tier.value,
            )
        entry["model"] = Prompt.ask(
            f"Model id for [bold]{tier.value}[/bold]",
            default=DEFAULT_MODELS[tier],
        )
        tiers[tier] = entry
    return tiers


def _prompt_reasoning(tiers: dict[Tier, dict[str, Any]]) -> None:
    """Optionally attach reasoning levels (extra_params) to specific tiers."""
    if not Confirm.ask(
        "Would you like to configure reasoning levels (extra_params) for any tier?",
        default=False,
    ):
        return

    while True:
        tier_key = Prompt.ask(
            "Tier to configure (mini/air/pro/ultra), or 'done' to finish",
            default="done",
        )
        if tier_key.lower() == "done":
            break
        try:
            tier = Tier(tier_key.lower())
        except ValueError:
            console.print(f"[red]Unknown tier '{tier_key}'[/red]")
            continue

        # Collect key-value pairs until the user is done with this tier.
        params: dict[str, Any] = {}
        while True:
            key = Prompt.ask(
                f"extra_param key for [bold]{tier.value}[/bold] (e.g. reasoning_effort), or 'done'",
                default="done",
            )
            if key.lower() == "done":
                break
            value = Prompt.ask(f"value for '{key}'")
            params[key] = value
        if params:
            tiers[tier]["extra_params"] = params


def _build_config(
    provider: dict[str, Any],
    tiers: dict[Tier, dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the full config dict from the collected answers."""
    provider_name = provider["name"]
    providers = {
        provider_name: {
            "base_url": provider["base_url"],
            "api_key_env": provider["api_key_env"],
        }
    }

    tier_cfg: dict[str, Any] = {}
    for tier in Tier:
        entry = tiers[tier]
        spec: dict[str, Any] = {
            "model": entry["model"],
            "name": entry["name"],
        }
        if entry.get("extra_params"):
            spec["extra_params"] = entry["extra_params"]
        tier_cfg[tier.value] = spec

    return {
        "default_tier": "air",
        "providers": providers,
        "tiers": tier_cfg,
        "classifier": {
            "model": "gemma4:31b",
            "provider": provider_name,
            "min_classify_len": 10,
        },
    }


def run_setup() -> None:
    """Run the interactive setup and write the config file."""
    console.print(
        Panel(
            "[bold]Axon Setup[/bold]\n"
            "Configure your model router: providers, tiers, and reasoning levels.",
            border_style="green",
        )
    )

    provider = _prompt_provider()
    tiers = _prompt_tiers()
    _prompt_reasoning(tiers)

    config = _build_config(provider, tiers)

    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))

    console.print(f"\n[green]✅ Config written to {path}[/green]")
    console.print("Run [bold]axon start[/bold] to launch the router.")
