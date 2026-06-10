from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        examples=["Fire spreading across California hills"],
    )
    keyword: str | None = None
    location: str | None = None


class BatchPredictRequest(BaseModel):
    tweets: list[PredictRequest] = Field(..., min_length=1, max_length=50)
