from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from src import config
from src.model_trainer import ensure_probabilities


def load_best_model(path: str | None = None):
    artifact_path = path if path else str(config.MODELS_DIR / "best_model.pkl")
    artifact = joblib.load(artifact_path)
    return artifact["model_name"], artifact["model"]


def predict_from_payload(payload: dict[str, Any], model_path: str | None = None) -> dict[str, Any]:
    model_name, model = load_best_model(model_path)
    frame = pd.DataFrame([payload])

    proba = float(ensure_probabilities(model, frame)[0])
    pred = int(proba >= 0.5)

    return {
        "model_name": model_name,
        "prediction": pred,
        "probability_canceled": round(proba, 6),
    }
