import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routes import health, predict
from src.config import settings
from src.utils.model_loader import load_model


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the model once at startup; release state on shutdown."""
    app.state.start_time = time.time()
    app.state.model_loaded = False
    app.state.artifact = None
    try:
        app.state.artifact = load_model(settings.MODEL_PATH)
        app.state.model_loaded = True
        logger.info("API ready | model={} port={}", settings.MODEL_VERSION, settings.API_PORT)
    except FileNotFoundError as exc:
        logger.error("Startup failed — model not found: {}", exc)
    yield
    app.state.artifact = None
    app.state.model_loaded = False
    logger.info("API shutdown complete.")


app = FastAPI(
    title="Disaster Tweets Detection API",
    version="1.0.0",
    description=(
        "Binary NLP classification: predict whether a tweet refers to a real disaster (1) "
        "or not (0). Model: NB_optuna — MultinomialNB + TF-IDF."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
