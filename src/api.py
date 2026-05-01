from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from src import config
from src.api_models import PredictRequest, TrainRequest
from src.predictor import predict_from_payload
from src.run_repository import get_run, list_runs, register_run
from trainer import run_pipeline


app = FastAPI(title="Hotel Booking Cancellation API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/train")
def train_model(payload: TrainRequest) -> dict[str, Any]:
    result = run_pipeline(
        quick_mode=payload.quick_mode,
        skip_neural_net=payload.skip_neural_net,
        enable_tuning=payload.enable_tuning,
        tuning_method=payload.tuning_method,
        tuning_cv=payload.tuning_cv,
        tuning_iter=payload.tuning_iter,
        enable_mlflow=payload.enable_mlflow,
    )
    metrics = result["metrics"].to_dict(orient="records")

    persisted = register_run(
        {
            "best_model": result["best_model"],
            "best_model_path": result["best_model_path"],
            "metrics_path": result["metrics_path"],
            "primary_metric": config.PRIMARY_METRIC,
            "primary_metric_value": float(result["metrics"].iloc[0][config.PRIMARY_METRIC]),
            "mlflow_run_id": result.get("mlflow_run_id"),
            "tuning": result.get("tuning"),
            "options": payload.model_dump(),
        }
    )

    return {
        "run_id": persisted["id"],
        "best_model": result["best_model"],
        "best_model_path": result["best_model_path"],
        "metrics_path": result["metrics_path"],
        "mlflow_run_id": result.get("mlflow_run_id"),
        "tuning": result.get("tuning"),
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


@app.get("/runs")
def runs(limit: int = 20) -> dict[str, Any]:
    safe_limit = max(1, min(limit, 200))
    results = list_runs(limit=safe_limit)
    return {
        "count": len(results),
        "results": results,
        "registry_file": str(config.RUNS_REGISTRY_PATH),
    }


@app.get("/runs/{run_id}")
def run_detail(run_id: str) -> dict[str, Any]:
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run
