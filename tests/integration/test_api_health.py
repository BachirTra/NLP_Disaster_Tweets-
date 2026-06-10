def test_root_returns_200(client) -> None:
    response = client.get("/")
    assert response.status_code == 200


def test_health_returns_200(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_health_model_loaded_true(client) -> None:
    response = client.get("/health")
    data = response.json()
    assert data["model_loaded"] is True
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert "model_version" in data
