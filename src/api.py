from __future__ import annotations

import json
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from src import config
from src.api_models import PredictRequest, TrainRequest
from src.predictor import load_best_model, predict_from_payload
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
        split_strategy=payload.split_strategy,
    )
    metrics = result["metrics"].to_dict(orient="records")

    persisted = register_run(
        {
            "best_model": result["best_model"],
            "best_model_path": result["best_model_path"],
            "metrics_path": result["metrics_path"],
            "primary_metric": config.PRIMARY_METRIC,
            "primary_metric_value": float(result["best_metrics"][config.PRIMARY_METRIC]),
            "mlflow_run_id": result.get("mlflow_run_id"),
            "tuning": result.get("tuning"),
            "prediction_threshold": result.get("prediction_threshold"),
            "threshold_selection_metric": result.get("threshold_selection_metric"),
            "calibration_method": result.get("calibration_method"),
            "resolved_split_strategy": result.get("resolved_split_strategy"),
            "serving_contract_path": result.get("serving_contract_path"),
            "monitoring_report_path": result.get("monitoring_report_path"),
            "explainability": result.get("explainability"),
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
        "prediction_threshold": result.get("prediction_threshold"),
        "threshold_selection_metric": result.get("threshold_selection_metric"),
        "calibration_method": result.get("calibration_method"),
        "resolved_split_strategy": result.get("resolved_split_strategy"),
        "serving_contract_path": result.get("serving_contract_path"),
        "monitoring_report_path": result.get("monitoring_report_path"),
        "explainability": result.get("explainability"),
        "metrics": metrics,
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    best_model_path = config.MODELS_DIR / "best_model.pkl"
    if not best_model_path.exists():
        raise HTTPException(status_code=400, detail="Model not found. Execute /train first.")

    return predict_from_payload(payload.features)


@app.get("/contract")
def contract() -> dict[str, Any]:
    best_model_path = config.MODELS_DIR / "best_model.pkl"
    if not best_model_path.exists():
        raise HTTPException(status_code=400, detail="Model not found. Execute /train first.")

    artifact = load_best_model()
    return {
        "model_name": artifact["model_name"],
        "artifact_version": artifact["artifact_version"],
        "split_strategy": artifact.get("split_strategy"),
        "contract": artifact.get("feature_contract"),
    }


@app.get("/monitoring")
def monitoring() -> dict[str, Any]:
    if not config.MONITORING_REPORT_PATH.exists():
        raise HTTPException(status_code=400, detail="Monitoring report not found. Execute /train first.")

    return json.loads(config.MONITORING_REPORT_PATH.read_text(encoding="utf-8"))


@app.get("/evaluate")
def evaluate() -> dict[str, Any]:
    if not config.METRICS_PATH.exists():
        raise HTTPException(status_code=400, detail="Metrics not found. Execute /train first.")

    df = pd.read_csv(config.METRICS_PATH)
    if "selected_for_deployment" in df.columns:
        selected_mask = df["selected_for_deployment"].astype(str).str.lower() == "true"
        selected_rows = df[selected_mask]
    else:
        selected_rows = df.iloc[0:0]
    best_model = (
        selected_rows.iloc[0]["model"]
        if not selected_rows.empty
        else df.sort_values(by=config.PRIMARY_METRIC, ascending=False).iloc[0]["model"]
    )
    return {
        "primary_metric": config.PRIMARY_METRIC,
        "best_model": best_model,
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
        "registry_file": str(config.RUNS_DB_PATH),
    }


@app.get("/runs/{run_id}")
def run_detail(run_id: str) -> dict[str, Any]:
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run
