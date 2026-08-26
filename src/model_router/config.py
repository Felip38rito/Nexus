"""Configuration for the router.

Settings are read from the environment (optionally from a `.env` file loaded
by the caller). See `.env.example`.
"""
from dataclasses import dataclass, field
import os
from pathlib import Path

import yaml

from .models import DEFAULT_TIER, ModelSpec, RouterModels, Tier


def _load_dotenv(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_models_yaml(path: Path | None) -> RouterModels | None:
    """Load a model table + classifier config from a YAML file.

    Returns None if the file is missing. Raises if the file is malformed or
    references an unknown tier. This lets the router's model set be configured
    without touching code.
    """
    if path is None or not path.exists():
        return None
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Malformed models YAML at {path}: expected a mapping")

    tiers: dict[Tier, ModelSpec] = {}
    raw_tiers = data.get("tiers") or {}
    if not isinstance(raw_tiers, dict):
        raise ValueError("'tiers' must be a mapping")
    for tier_key, spec in raw_tiers.items():
        try:
            tier = Tier(str(tier_key))
        except ValueError:
            raise ValueError(f"Unknown tier '{tier_key}' in {path}")
        if not isinstance(spec, dict) or not spec.get("model"):
            raise ValueError(f"Tier '{tier_key}' must define a 'model'")
        tiers[tier] = ModelSpec(
            api_id=str(spec["model"]),
            description=str(spec.get("description", "")),
        )

    # Require all four tiers.
    for tier in Tier:
        if tier not in tiers:
            raise ValueError(f"Missing tier '{tier.value}' in models YAML {path}")

    classifier = data.get("classifier") or {}
    classifier_model = str(classifier.get("model", "gemma4:31b"))
    min_classify_len = int(classifier.get("min_classify_len", 20))

    default_raw = data.get("default_tier")
    try:
        default_tier = Tier(str(default_raw)) if default_raw else DEFAULT_TIER
    except ValueError:
        raise ValueError(f"Unknown default_tier '{default_raw}' in {path}")

    return RouterModels(
        tiers=tiers,
        default_tier=default_tier,
        classifier_model=classifier_model,
        min_classify_len=min_classify_len,
    )


@dataclass
class Settings:
    ollama_api_key: str
    ollama_base_url: str = "https://ollama.com/v1"
    router_host: str = "127.0.0.1"
    router_port: int = 8000
    default_tier: Tier = DEFAULT_TIER
    # Minimum combined message length before we bother classifying at all.
    # Below this, the request is treated as trivial (mini).
    min_classify_len: int = 20
    # Optional bearer token clients must send to reach the router.
    require_auth: str = ""
    # Mounted model table + classifier config (from YAML or defaults).
    models: RouterModels = field(default_factory=RouterModels)

    @classmethod
    def from_env(
        cls,
        dotenv_path: Path | None = None,
        default_models_yaml: Path | None = None,
    ) -> "Settings":
        _load_dotenv(dotenv_path)
        key = os.environ.get("OLLAMA_API_KEY", "").strip()
        require_auth = os.environ.get("ROUTER_API_KEY", "").strip()

        # Load models YAML: explicit ROUTER_MODELS_YAML wins, else the default
        # project file, else built-in defaults.
        yaml_path_raw = os.environ.get("ROUTER_MODELS_YAML", "").strip()
        if yaml_path_raw:
            yaml_path: Path | None = Path(yaml_path_raw)
        else:
            yaml_path = default_models_yaml
        models = load_models_yaml(yaml_path) or RouterModels()

        return cls(
            ollama_api_key=key,
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/v1").rstrip("/"),
            router_host=os.environ.get("ROUTER_HOST", "127.0.0.1"),
            router_port=int(os.environ.get("ROUTER_PORT", "8000")),
            default_tier=Tier(os.environ.get("ROUTER_DEFAULT_TIER", DEFAULT_TIER.value)),
            min_classify_len=int(os.environ.get("ROUTER_MIN_CLASSIFY_LEN", "20")),
            require_auth=require_auth,
            models=models,
        )

    @property
    def effective_api_key(self) -> str:
        """Raise early if the upstream key is missing."""
        if not self.ollama_api_key:
            raise RuntimeError("OLLAMA_API_KEY is not set")
        return self.ollama_api_key
