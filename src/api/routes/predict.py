import time

from fastapi import APIRouter, Request
from loguru import logger

from src.api.schemas.request import BatchPredictRequest, PredictRequest
from src.api.schemas.response import BatchPredictResponse, PredictResponse
from src.utils.model_loader import predict_single

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictResponse, summary="Single tweet prediction")
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    """Classify a single tweet as disaster or not.

    Args:
        body: Request payload containing the tweet text.
        request: FastAPI request object (provides access to app state).

    Returns:
        PredictResponse with label, probabilities, and inference timing.
    """
    artifact = request.app.state.artifact
    result = predict_single(body.text, artifact)
    logger.info(
        "predict | input_length={} prediction={} label={} latency_ms={}",
        len(body.text),
        result.prediction,
        result.label,
        result.inference_time_ms,
    )
    return result


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Batch tweet prediction",
)
def predict_batch(body: BatchPredictRequest, request: Request) -> BatchPredictResponse:
    """Classify up to 50 tweets in a single call.

    Args:
        body: Request payload with a list of tweet objects.
        request: FastAPI request object (provides access to app state).

    Returns:
        BatchPredictResponse with per-tweet results and total timing.
    """
    t0 = time.perf_counter()
    artifact = request.app.state.artifact
    results = [predict_single(tweet.text, artifact) for tweet in body.tweets]
    total_ms = round((time.perf_counter() - t0) * 1_000, 3)
    logger.info(
        "batch_predict | count={} total_latency_ms={}",
        len(body.tweets),
        total_ms,
    )
    return BatchPredictResponse(results=results, total=len(results), total_time_ms=total_ms)
