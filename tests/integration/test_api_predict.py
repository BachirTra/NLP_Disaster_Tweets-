_REQUIRED_FIELDS = {
    "prediction",
    "label",
    "probability_disaster",
    "probability_not_disaster",
    "model_version",
    "inference_time_ms",
}


def test_predict_disaster_tweet(client, sample_disaster_tweet) -> None:
    response = client.post("/predict", json={"text": sample_disaster_tweet})
    assert response.status_code == 200
    assert response.json()["prediction"] == 1


def test_predict_non_disaster_tweet(client, sample_non_disaster_tweet) -> None:
    response = client.post("/predict", json={"text": sample_non_disaster_tweet})
    assert response.status_code == 200
    assert response.json()["prediction"] == 0


def test_predict_response_has_required_fields(client, sample_disaster_tweet) -> None:
    response = client.post("/predict", json={"text": sample_disaster_tweet})
    assert response.status_code == 200
    data = response.json()
    for field in _REQUIRED_FIELDS:
        assert field in data, f"Missing field: {field}"


def test_predict_empty_text_returns_422(client) -> None:
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_text_too_long_returns_422(client) -> None:
    response = client.post("/predict", json={"text": "a" * 501})
    assert response.status_code == 422


def test_batch_predict_returns_correct_count(
    client, sample_disaster_tweet, sample_non_disaster_tweet
) -> None:
    payload = {
        "tweets": [
            {"text": sample_disaster_tweet},
            {"text": sample_non_disaster_tweet},
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["results"]) == 2
