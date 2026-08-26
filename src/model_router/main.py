"""FastAPI app assembly for the model router."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI

from . import __version__
from .config import Settings
from .proxy import router as proxy_router


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="Nexus",
        version=__version__,
        description="Local OpenAI-compatible proxy routing requests to the cheapest adequate Ollama Cloud model.",
    )

    if settings is None:
        project_root = Path(__file__).resolve().parents[2]
        settings = Settings.from_env(
            project_root / ".env",
            default_models_yaml=project_root / "router.models.yaml",
        )
        settings.project_root = project_root

    app.state.settings = settings
    app.include_router(proxy_router)
    return app


app = create_app()
