from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from src import config


class MLflowTracker:
    """Encapsula el registro en MLflow usado por la canalización de entrenamiento.

    Al crear una instancia se configuran la URI de seguimiento y el experimento
    globales a partir de :mod:`src.config`.
    """

    def __init__(self) -> None:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    def start_run(self, run_name: str, tags: dict[str, str] | None = None):
        """Inicia y devuelve un nuevo contexto de ejecución de MLflow.

        :param run_name: Nombre visible de la ejecución.
        :param tags: Etiquetas opcionales asociadas al run.
        :returns: Context manager devuelto por :func:`mlflow.start_run`.
        """
        return mlflow.start_run(run_name=run_name, tags=tags)

    @staticmethod
    def log_params(params: dict[str, Any]) -> None:
        """Registra un lote de parámetros en la ejecución activa de MLflow.

        :param params: Diccionario de parámetros serializables.
        """
        mlflow.log_params(params)

    @staticmethod
    def log_metrics(metrics: dict[str, float]) -> None:
        """Registra un lote de métricas en la ejecución activa de MLflow.

        :param metrics: Diccionario ``nombre -> valor`` de métricas numéricas.
        """
        mlflow.log_metrics(metrics)

    @staticmethod
    def log_artifact(path: str | Path) -> None:
        """Sube un artefacto local a la ejecución activa de MLflow.

        :param path: Ruta del fichero a registrar.
        """
        mlflow.log_artifact(str(path))

    @staticmethod
    def log_model(model, input_example=None, artifact_path: str = "model") -> None:
        """Registra un modelo de scikit-learn e infiere su firma cuando es posible.

        Las columnas enteras de un ``input_example`` basado en pandas se
        convierten a ``float64`` antes de inferir la firma para mantener la
        compatibilidad con MLflow.

        :param model: Modelo entrenado que se va a registrar.
        :param input_example: Ejemplo opcional de entrada para inferir la firma.
        :param artifact_path: Nombre de la carpeta de artefactos en MLflow.
        """
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
