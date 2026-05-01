from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from src import config
from src.api_models import PredictRequest, TrainRequest
from src.predictor import predict_from_payload
from trainer import run_pipeline


app = FastAPI(title="Hotel Booking Cancellation API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/train")
def train_model(payload: TrainRequest) -> dict[str, Any]:
    result = run_pipeline(quick_mode=payload.quick_mode, skip_neural_net=payload.skip_neural_net)
    metrics = result["metrics"].to_dict(orient="records")

    return {
        "best_model": result["best_model"],
        "best_model_path": result["best_model_path"],
        "metrics_path": result["metrics_path"],
        "metrics": metrics,
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    best_model_path = config.MODELS_DIR / "best_model.pkl"
    if not best_model_path.exists():
        raise HTTPException(status_code=400, detail="Model not found. Execute /train first.")

    return predict_from_payload(payload.features)


@app.get("/evaluate")
def evaluate() -> dict[str, Any]:
    if not config.METRICS_PATH.exists():
        raise HTTPException(status_code=400, detail="Metrics not found. Execute /train first.")

    df = pd.read_csv(config.METRICS_PATH)
    return {
        "primary_metric": config.PRIMARY_METRIC,
        "best_model": df.sort_values(by=config.PRIMARY_METRIC, ascending=False).iloc[0]["model"],
        "results": df.to_dict(orient="records"),
        "metrics_file": str(config.METRICS_PATH),
    }
