import time

from fastapi import APIRouter, Request
from loguru import logger

from src.api.schemas.response import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/", summary="Project info")
def root() -> dict:
    """Return basic project metadata."""
    return {
        "name": "Disaster Tweets Detection API",
        "version": "1.0.0",
        "description": "Binary NLP classification: predict whether a tweet refers to a real disaster.",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health(request: Request) -> HealthResponse:
    """Return API and model liveness status."""
    uptime = round(time.time() - request.app.state.start_time, 2)
    model_loaded: bool = getattr(request.app.state, "model_loaded", False)
    artifact = getattr(request.app.state, "artifact", None)
    model_version = (
        str(artifact.get("model_name", "unknown")) if artifact else "not_loaded"
    )
    logger.debug("health check | model_loaded={} uptime={}s", model_loaded, uptime)
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        model_version=model_version,
        uptime_seconds=uptime,
    )
