from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    quick_mode: bool = Field(default=False)
    skip_neural_net: bool = Field(default=False)


class PredictRequest(BaseModel):
    features: dict[str, Any]
