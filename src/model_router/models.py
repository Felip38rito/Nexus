"""Model tier table: tier key -> model spec.

The default table is the LIVE `/v1/models` API ids (verified 2026-08-25). It
can be overridden at runtime via a YAML config file (see `router.models.yaml`).
IMPORTANT: use raw API ids — the `:cloud` suffix is an Hermes alias only and
returns 404 against the Ollama Cloud API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Tier(str, Enum):
    MINI = "mini"
    AIR = "air"
    PRO = "pro"
    ULTRA = "ultra"


@dataclass(frozen=True)
class ProviderSpec:
    """A named upstream endpoint a tier (or the classifier) can point at.

    ``base_url`` is the provider's OpenAI-compatible ``/v1`` endpoint.
    ``api_key`` is the key provided inline.
    ``api_key_env`` is the environment variable holding that provider's key.
    """

    base_url: str
    api_key: str | None = None
    api_key_env: str | None = "OLLAMA_API_KEY"

    def resolve_api_key(self) -> str:
        """Resolve the API key from inline value or environment variable.
        
        Raises RuntimeError if neither is provided or if both are set (ambiguity).
        """
        import os
        inline = self.api_key
        env_var = self.api_key_env
        env_val = os.environ.get(env_var) if env_var else None

        if inline and env_val:
            raise RuntimeError(f"Ambiguous API key for provider: both inline and env var '{env_var}' are set.")
        
        if env_val:
            return env_val
        if inline:
            return inline
            
        raise RuntimeError(f"No API key found for provider: inline is empty and env var '{env_var}' is not set.")


@dataclass(frozen=True)
class ModelSpec:
    api_id: str
    description: str
    # Which named provider (see Settings.providers) serves this tier. Defaults
    # to "default" so existing single-provider configs keep working unchanged.
    provider: str = "default"
    # Optional display/route alias. If set, /v1/models advertises this as the
    # model id and the proxy accepts it as an alias for the tier. If None, the
    # tier key (mini/air/pro/ultra) is used. Never affects the classifier's
    # internal key.
    name: str | None = None


# The built-in single-provider table. `air` is the default for day-to-day work.
_DEFAULT_TABLE: dict[Tier, ModelSpec] = {
    Tier.MINI: ModelSpec("gemma4:31b", "fast/cheap — discussion + trivial/mechanical tasks"),
    Tier.AIR: ModelSpec("deepseek-v4-flash:0731", "default — day-to-day implementation"),
    Tier.PRO: ModelSpec("deepseek-v4-pro:0813", "raw coding power, hard debugging/refactors"),
    Tier.ULTRA: ModelSpec("kimi-k3", "deep synthesis, whole-architecture, adversarial analysis"),
}

# The default provider every tier uses when no `providers:` block is configured.
DEFAULT_PROVIDERS: dict[str, ProviderSpec] = {
    "default": ProviderSpec(
        base_url="https://ollama.com/v1",
        api_key_env="OLLAMA_API_KEY",
    ),
}

DEFAULT_TIER = Tier.AIR


@dataclass(frozen=True)
class RouterModels:
    """Mounted model table + classifier config for the running router."""

    tiers: dict[Tier, ModelSpec] = field(default_factory=lambda: dict(_DEFAULT_TABLE))
    default_tier: Tier = DEFAULT_TIER
    classifier_model: str = "gemma4:31b"
    # Which named provider serves the classifier (defaults to "default").
    classifier_provider: str = "default"
    min_classify_len: int = 10
    # Named upstream endpoints. Each tier's ModelSpec.provider keys into this.
    providers: dict[str, ProviderSpec] = field(default_factory=lambda: dict(DEFAULT_PROVIDERS))

    def tier_for_api_id(self, api_id: str) -> Tier | None:
        for tier, spec in self.tiers.items():
            if spec.api_id == api_id:
                return tier
        return None

    def provider_for(self, provider_name: str) -> ProviderSpec:
        """Resolve a provider name to its spec, falling back to "default"."""
        return self.providers.get(provider_name) or self.providers.get("default", DEFAULT_PROVIDERS["default"])

    @property
    def api_ids(self) -> list[str]:
        return [spec.api_id for spec in self.tiers.values()]


# Backwards-compatible module-level convenience (used by tests + callers that
# don't have a settings object). Prefer passing a RouterModels instance.
MODEL_TABLE: dict[Tier, ModelSpec] = dict(_DEFAULT_TABLE)


def tier_for_api_id(api_id: str) -> Tier | None:
    return RouterModels().tier_for_api_id(api_id)
