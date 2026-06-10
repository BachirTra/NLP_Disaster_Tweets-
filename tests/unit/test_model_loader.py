import pytest

from src.utils.model_loader import load_model

_MODEL_PATH = "models/NB_optuna.pkl"
_REQUIRED_KEYS = {"model", "tfidf", "model_name", "f1_val", "f1_test"}


def test_load_model_returns_dict() -> None:
    artifact = load_model(_MODEL_PATH)
    assert isinstance(artifact, dict)


def test_load_model_has_required_keys() -> None:
    artifact = load_model(_MODEL_PATH)
    for key in ("model", "tfidf", "f1_test"):
        assert key in artifact, f"Missing key: {key}"


def test_load_model_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_model("models/nonexistent_model.pkl")
