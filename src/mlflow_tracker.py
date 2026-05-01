from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from src import config


class MLflowTracker:
    def __init__(self) -> None:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    def start_run(self, run_name: str, tags: dict[str, str] | None = None):
        return mlflow.start_run(run_name=run_name, tags=tags)

    @staticmethod
    def log_params(params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    @staticmethod
    def log_metrics(metrics: dict[str, float]) -> None:
        mlflow.log_metrics(metrics)

    @staticmethod
    def log_artifact(path: str | Path) -> None:
        mlflow.log_artifact(str(path))

    @staticmethod
    def log_model(model, artifact_path: str = "model") -> None:
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)
