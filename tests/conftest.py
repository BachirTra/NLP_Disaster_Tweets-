import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient with full lifespan (model loaded)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_disaster_tweet() -> str:
    return "Massive earthquake hits Turkey, thousands feared dead"


@pytest.fixture
def sample_non_disaster_tweet() -> str:
    return "I'm on fire today at the gym!"
