"""FastAPI app assembly for the model router."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

import httpx
from fastapi import FastAPI

from . import __version__
from .config import Settings
from .proxy import router as proxy_router

log = logging.getLogger("model_router")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # One shared HTTP client for all upstream calls (reused, closed on shutdown).
    app.state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=15.0))
    try:
        yield
    finally:
        await app.state.http_client.aclose()


def _setup_logging(project_root: Path) -> None:
    root = logging.getLogger("model_router")
    if root.handlers:
        return  # already configured (e.g. tests reuse create_app)
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root.addHandler(stream)

    file_handler = RotatingFileHandler(
        project_root / "router.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="Axon",
        version=__version__,
        description="Local OpenAI-compatible proxy routing requests to the cheapest adequate Ollama Cloud model.",
        lifespan=_lifespan,
    )

    if settings is None:
        project_root = Path(__file__).resolve().parents[2]
        settings = Settings.from_env(
            project_root / ".env",
            default_models_yaml=project_root / "router.models.yaml",
        )
        settings.project_root = project_root
        _setup_logging(project_root)

    app.state.settings = settings
    app.include_router(proxy_router)
    return app


app = create_app()
