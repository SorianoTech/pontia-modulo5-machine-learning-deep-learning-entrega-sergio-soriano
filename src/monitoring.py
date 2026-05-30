from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.model_trainer import ensure_probabilities


def write_json_artifact(payload: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def build_serving_contract(X_reference: pd.DataFrame) -> dict[str, Any]:
    features = []
    for column_name, dtype in X_reference.dtypes.items():
        kind = "numeric" if pd.api.types.is_numeric_dtype(dtype) else "categorical"
        features.append(
            {
                "name": column_name,
                "dtype": str(dtype),
                "kind": kind,
                "required": True,
            }
        )

    return {
        "contract_version": 1,
        "feature_count": len(features),
        "required_features": [feature["name"] for feature in features],
        "allowed_extra_features": False,
        "features": features,
        "prediction_fields": [
            "model_name",
            "prediction",
            "probability_canceled",
            "prediction_threshold",
            "threshold_selection_metric",
            "calibration_method",
        ],
    }


def validate_payload_against_contract(payload: dict[str, Any], contract: dict[str, Any] | None) -> dict[str, Any]:
    if not contract:
        return payload

    required_features = contract.get("required_features", [])
    feature_specs = {feature["name"]: feature for feature in contract.get("features", [])}

    payload_keys = set(payload.keys())
    required_keys = set(required_features)
    missing = sorted(required_keys - payload_keys)
    unexpected = sorted(payload_keys - required_keys)

    if missing:
        raise ValueError(f"Faltan features requeridas: {', '.join(missing)}")
    if unexpected and not contract.get("allowed_extra_features", False):
        raise ValueError(f"Features no esperadas: {', '.join(unexpected)}")

    ordered_payload: dict[str, Any] = {}
    for feature_name in required_features:
        value = payload[feature_name]
        spec = feature_specs.get(feature_name, {})
        if value is not None:
            if spec.get("kind") == "numeric" and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ValueError(f"La feature '{feature_name}' debe ser numérica o null.")
            if spec.get("kind") == "categorical" and not isinstance(value, str):
                raise ValueError(f"La feature '{feature_name}' debe ser texto o null.")
        ordered_payload[feature_name] = value
    return ordered_payload


def _score_summary(probabilities: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(probabilities)),
        "std": float(np.std(probabilities)),
        "min": float(np.min(probabilities)),
        "p05": float(np.quantile(probabilities, 0.05)),
        "p50": float(np.quantile(probabilities, 0.5)),
        "p95": float(np.quantile(probabilities, 0.95)),
        "max": float(np.max(probabilities)),
    }


def _numeric_drift(train_series: pd.Series, test_series: pd.Series) -> dict[str, Any]:
    train_std = float(train_series.std()) if float(train_series.std()) > 0 else 1.0
    drift_score = abs(float(test_series.mean()) - float(train_series.mean())) / train_std
    return {
        "kind": "numeric",
        "drift_score": float(drift_score),
        "train_missing_rate": float(train_series.isna().mean()),
        "test_missing_rate": float(test_series.isna().mean()),
        "train_mean": float(train_series.mean()),
        "test_mean": float(test_series.mean()),
    }


def _categorical_drift(train_series: pd.Series, test_series: pd.Series) -> dict[str, Any]:
    train_dist = train_series.fillna("<<NULL>>").astype(str).value_counts(normalize=True)
    test_dist = test_series.fillna("<<NULL>>").astype(str).value_counts(normalize=True)
    all_categories = sorted(set(train_dist.index).union(test_dist.index))
    drift_score = 0.5 * sum(abs(float(train_dist.get(cat, 0.0)) - float(test_dist.get(cat, 0.0))) for cat in all_categories)
    return {
        "kind": "categorical",
        "drift_score": float(drift_score),
        "train_missing_rate": float(train_series.isna().mean()),
        "test_missing_rate": float(test_series.isna().mean()),
        "top_test_only_categories": [cat for cat in all_categories if cat not in set(train_dist.index)][:5],
    }


def build_monitoring_report(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model,
    split_metadata: dict[str, Any],
) -> dict[str, Any]:
    train_scores = ensure_probabilities(model, X_train)
    test_scores = ensure_probabilities(model, X_test)

    drift_rows = []
    for column in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[column]):
            feature_report = _numeric_drift(X_train[column], X_test[column])
        else:
            feature_report = _categorical_drift(X_train[column], X_test[column])
        drift_rows.append({"feature": column, **feature_report})

    top_drift = sorted(drift_rows, key=lambda row: row["drift_score"], reverse=True)[: config.DRIFT_TOP_FEATURES]

    return {
        "split_strategy": split_metadata.get("split_strategy"),
        "train_period_start": split_metadata.get("train_period_start"),
        "train_period_end": split_metadata.get("train_period_end"),
        "test_period_start": split_metadata.get("test_period_start"),
        "test_period_end": split_metadata.get("test_period_end"),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_positive_rate": float(np.mean(y_train)),
        "test_positive_rate": float(np.mean(y_test)),
        "score_summary": {
            "train": _score_summary(train_scores),
            "test": _score_summary(test_scores),
        },
        "top_feature_drift": top_drift,
    }
