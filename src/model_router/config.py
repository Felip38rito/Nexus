"""Configuration for the router.

Settings are read from the environment (optionally from a `.env` file loaded
by the caller). See `.env.example`.
"""
from dataclasses import dataclass, field
import os
from pathlib import Path

import yaml

from .models import (
    DEFAULT_PROVIDERS,
    DEFAULT_TIER,
    ModelSpec,
    ProviderSpec,
    RouterModels,
    Tier,
    _DEFAULT_TABLE,
)


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
    references an unknown tier/provider. This lets the router's model set be
    configured without touching code.

    The YAML may define a ``providers:`` block mapping a provider name to its
    ``base_url`` and ``api_key_env``. Each tier (and the classifier) can then
    reference one of those providers via ``provider: <name>``. If no
    ``providers:`` block is given, a single ``default`` provider (Ollama Cloud)
    is used and every tier points at it — preserving the original behavior.
    """
    if path is None or not path.exists():
        return None
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Malformed models YAML at {path}: expected a mapping")

    # --- Providers ---
    providers: dict[str, ProviderSpec] = dict(DEFAULT_PROVIDERS)
    raw_providers = data.get("providers") or {}
    if raw_providers:
        if not isinstance(raw_providers, dict):
            raise ValueError("'providers' must be a mapping")
        providers = {}
        for name, pcfg in raw_providers.items():
            if not isinstance(pcfg, dict) or not pcfg.get("base_url"):
                raise ValueError(f"Provider '{name}' must define a 'base_url'")
            providers[str(name)] = ProviderSpec(
                base_url=str(pcfg["base_url"]).rstrip("/"),
                api_key=str(pcfg.get("api_key")),
                api_key_env=str(pcfg.get("api_key_env", "OLLAMA_API_KEY")) if "api_key_env" in pcfg else None,
            )
        # Always ensure a "default" provider exists so tiers that don't specify
        # one (or the classifier) still resolve.
        providers.setdefault("default", DEFAULT_PROVIDERS["default"])

    # --- Tiers ---
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
        provider = str(spec.get("provider", "default"))
        if provider not in providers:
            raise ValueError(
                f"Tier '{tier_key}' references unknown provider '{provider}' in {path}"
            )
        tiers[tier] = ModelSpec(
            api_id=str(spec["model"]),
            description=str(spec.get("description", _DEFAULT_TABLE[tier].description)),
            provider=provider,
            name=str(spec["name"]) if spec.get("name") else None,
        )

    # Require all four tiers.
    for tier in Tier:
        if tier not in tiers:
            raise ValueError(f"Missing tier '{tier.value}' in models YAML {path}")

    # --- Classifier ---
    classifier = data.get("classifier") or {}
    classifier_model = str(classifier.get("model", "gemma4:31b"))
    classifier_provider = str(classifier.get("provider", "default"))
    if classifier_provider not in providers:
        raise ValueError(
            f"Classifier references unknown provider '{classifier_provider}' in {path}"
        )
    min_classify_len = int(classifier.get("min_classify_len", 10))

    default_raw = data.get("default_tier")
    try:
        default_tier = Tier(str(default_raw)) if default_raw else DEFAULT_TIER
    except ValueError:
        raise ValueError(f"Unknown default_tier '{default_raw}' in {path}")

    return RouterModels(
        tiers=tiers,
        default_tier=default_tier,
        classifier_model=classifier_model,
        classifier_provider=classifier_provider,
        min_classify_len=min_classify_len,
        providers=providers,
    )


@dataclass
class Settings:
    ollama_api_key: str
    ollama_base_url: str = "https://ollama.com/v1"
    router_host: str = "127.0.0.1"
    router_port: int = 9000
    default_tier: Tier = DEFAULT_TIER
    # Minimum combined message length before we bother classifying at all.
    # Below this, the request is treated as trivial (mini).
    min_classify_len: int = 10
    # Optional bearer token clients must send to reach the router.
    require_auth: str = ""
    # Mounted model table + classifier config (from YAML or defaults).
    models: RouterModels = field(default_factory=RouterModels)
    # Where router.log is written. Set by create_app; defaults to the repo root.
    project_root: Path = field(default_factory=lambda: Path.cwd())

    @classmethod
    def from_env(
        cls,
        dotenv_path: Path | None = None,
        default_models_yaml: Path | None = None,
    ) -> "Settings":
        _load_dotenv(dotenv_path)
        key = os.environ.get("OLLAMA_API_KEY", "").strip()
        require_auth = os.environ.get("ROUTER_API_KEY", "").strip()

        # Load models YAML: explicit ROUTER_MODELS_YAML wins, else the
        # user config in ~/.config/axon/config.yml, else the project
        # default file, else built-in defaults.
        yaml_path_raw = os.environ.get("ROUTER_MODELS_YAML", "").strip()
        if yaml_path_raw:
            yaml_path: Path | None = Path(yaml_path_raw)
        else:
            user_cfg = Path.home() / ".config" / "axon" / "config.yml"
            if user_cfg.exists():
                yaml_path = user_cfg
            else:
                yaml_path = default_models_yaml
        
        models = load_models_yaml(yaml_path) or RouterModels()

        return cls(
            ollama_api_key=key,
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/v1").rstrip("/"),
            router_host=os.environ.get("ROUTER_HOST", "127.0.0.1"),
            router_port=int(os.environ.get("ROUTER_PORT", "9000")),
            default_tier=Tier(os.environ.get("ROUTER_DEFAULT_TIER", DEFAULT_TIER.value)),
            min_classify_len=int(os.environ.get("ROUTER_MIN_CLASSIFY_LEN", "10")),
            require_auth=require_auth,
            models=models,
        )

    @property
    def effective_api_key(self) -> str:
        """Raise early if the upstream key is missing."""
        if not self.ollama_api_key:
            raise RuntimeError("OLLAMA_API_KEY is not set")
        return self.ollama_api_key
