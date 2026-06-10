import time
from pathlib import Path

import dill
import numpy as np
from loguru import logger

from src.api.schemas.response import PredictResponse
from src.data.preprocess import preprocess_data


def load_model(path: str) -> dict:
    """Load the serialised model artifact from disk.

    Args:
        path: Relative or absolute path to the .pkl artifact.

    Returns:
        dict with keys: model, tfidf, scaler, model_name, features, f1_val, f1_test.

    Raises:
        FileNotFoundError: When the artifact file does not exist at *path*.
    """
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    with model_path.open("rb") as f:
        artifact = dill.load(f)

    logger.info(
        "Model loaded | version={} f1_val={:.4f} f1_test={:.4f}",
        artifact.get("model_name", "unknown"),
        artifact.get("f1_val", 0.0),
        artifact.get("f1_test", 0.0),
    )
    return artifact


def predict_single(text: str, artifact: dict) -> PredictResponse:
    """Run single-tweet inference through the NB pipeline.

    Args:
        text: Raw tweet text (not yet preprocessed).
        artifact: Model artifact returned by :func:`load_model`.

    Returns:
        PredictResponse with prediction, probabilities, and wall-clock timing.
    """
    t0 = time.perf_counter()

    processed = preprocess_data(text)
    X = artifact["tfidf"].transform([processed])
    proba = artifact["model"].predict_proba(X)[0]
    prediction = int(np.argmax(proba))

    inference_ms = round((time.perf_counter() - t0) * 1_000, 3)

    return PredictResponse(
        prediction=prediction,
        label="disaster" if prediction == 1 else "not_disaster",
        probability_disaster=round(float(proba[1]), 4),
        probability_not_disaster=round(float(proba[0]), 4),
        model_version=str(artifact.get("model_name", "unknown")),
        inference_time_ms=inference_ms,
    )
