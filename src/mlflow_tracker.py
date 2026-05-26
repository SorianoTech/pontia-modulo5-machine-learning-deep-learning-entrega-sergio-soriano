from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import infer_signature

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
    def log_model(model, input_example=None, artifact_path: str = "model") -> None:
        prepared_input_example = input_example
        if isinstance(input_example, pd.DataFrame):
            prepared_input_example = input_example.copy()
            integer_columns = prepared_input_example.select_dtypes(include=["integer"]).columns
            if len(integer_columns) > 0:
                prepared_input_example[integer_columns] = prepared_input_example[integer_columns].astype("float64")

        signature = None
        if prepared_input_example is not None:
            signature = infer_signature(
                prepared_input_example,
                model.predict(prepared_input_example),
            )

        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            input_example=prepared_input_example,
            signature=signature,
        )
