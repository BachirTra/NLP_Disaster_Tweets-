from pydantic import BaseModel


class PredictResponse(BaseModel):
    prediction: int
    label: str
    probability_disaster: float
    probability_not_disaster: float
    model_version: str
    inference_time_ms: float


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    total: int
    total_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
