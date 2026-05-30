from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from src import config
from src.monitoring import validate_payload_against_contract
from src.model_trainer import ensure_probabilities


def load_best_model(path: str | None = None):
    artifact_path = path if path else str(config.MODELS_DIR / "best_model.pkl")
    artifact = joblib.load(artifact_path)
    return {
        "model_name": artifact["model_name"],
        "model": artifact["model"],
        "prediction_threshold": float(
            artifact.get("prediction_threshold", config.DEFAULT_PREDICTION_THRESHOLD)
        ),
        "artifact_version": int(artifact.get("artifact_version", 1)),
        "threshold_selection_metric": artifact.get("threshold_selection_metric"),
        "calibration_method": artifact.get("calibration_method"),
        "split_strategy": artifact.get("split_strategy"),
        "feature_contract": artifact.get("feature_contract"),
        "monitoring_report": artifact.get("monitoring_report"),
        "explainability": artifact.get("explainability"),
    }


def predict_from_payload(payload: dict[str, Any], model_path: str | None = None) -> dict[str, Any]:
    artifact = load_best_model(model_path)
    validated_payload = validate_payload_against_contract(payload, artifact.get("feature_contract"))
    frame = pd.DataFrame([validated_payload])

    proba = float(ensure_probabilities(artifact["model"], frame)[0])
    pred = int(proba >= artifact["prediction_threshold"])

    return {
        "model_name": artifact["model_name"],
        "prediction": pred,
        "probability_canceled": round(proba, 6),
        "prediction_threshold": round(artifact["prediction_threshold"], 6),
        "threshold_selection_metric": artifact["threshold_selection_metric"],
        "calibration_method": artifact["calibration_method"],
        "split_strategy": artifact["split_strategy"],
        "artifact_version": artifact["artifact_version"],
    }
