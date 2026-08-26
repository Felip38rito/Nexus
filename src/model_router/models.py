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
    PRO_MAX = "pro-max"


@dataclass(frozen=True)
class ModelSpec:
    api_id: str
    description: str


# Ordered cheapest -> most capable. `air` is the default for day-to-day work.
_DEFAULT_TABLE: dict[Tier, ModelSpec] = {
    Tier.MINI: ModelSpec("gemma4:31b", "fast/cheap — discussion + trivial/mechanical tasks"),
    Tier.AIR: ModelSpec("deepseek-v4-flash:0731", "default — day-to-day implementation"),
    Tier.PRO: ModelSpec("glm-5.2", "frontier reasoning, complex/ambiguous design"),
    Tier.PRO_MAX: ModelSpec("deepseek-v4-pro:0813", "raw coding power, hard debugging/refactors"),
}

DEFAULT_TIER = Tier.AIR


@dataclass(frozen=True)
class RouterModels:
    """Mounted model table + classifier config for the running router."""

    tiers: dict[Tier, ModelSpec] = field(default_factory=lambda: dict(_DEFAULT_TABLE))
    default_tier: Tier = DEFAULT_TIER
    classifier_model: str = "gemma4:31b"
    min_classify_len: int = 20

    def tier_for_api_id(self, api_id: str) -> Tier | None:
        for tier, spec in self.tiers.items():
            if spec.api_id == api_id:
                return tier
        return None

    @property
    def api_ids(self) -> list[str]:
        return [spec.api_id for spec in self.tiers.values()]


# Backwards-compatible module-level convenience (used by tests + callers that
# don't have a settings object). Prefer passing a RouterModels instance.
MODEL_TABLE: dict[Tier, ModelSpec] = dict(_DEFAULT_TABLE)


def tier_for_api_id(api_id: str) -> Tier | None:
    return RouterModels().tier_for_api_id(api_id)
