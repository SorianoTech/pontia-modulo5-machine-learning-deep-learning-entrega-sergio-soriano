from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    quick_mode: bool = Field(default=False)
    skip_neural_net: bool = Field(default=False)
    enable_tuning: bool = Field(default=False)
    tuning_method: str = Field(default="randomized")
    tuning_cv: int = Field(default=3, ge=2)
    tuning_iter: int = Field(default=15, ge=1)
    enable_mlflow: bool = Field(default=True)


class PredictRequest(BaseModel):
    features: dict[str, Any]
