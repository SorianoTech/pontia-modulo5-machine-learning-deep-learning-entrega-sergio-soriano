from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    """Esquema de entrada para lanzar un entrenamiento desde la API.

    :param quick_mode: Activa una versión reducida del entrenamiento.
    :param skip_neural_net: Omite el modelo basado en red neuronal.
    :param enable_tuning: Habilita el ajuste de hiperparámetros del ganador.
    :param tuning_method: Método de búsqueda para el tuning.
    :param tuning_cv: Número de folds usados en validación cruzada.
    :param tuning_iter: Iteraciones a ejecutar en la búsqueda aleatoria.
    :param enable_mlflow: Indica si se debe registrar la ejecución en MLflow.
    """
    quick_mode: bool = Field(default=False)
    skip_neural_net: bool = Field(default=False)
    enable_tuning: bool = Field(default=False)
    tuning_method: str = Field(default="randomized")
    tuning_cv: int = Field(default=3, ge=2)
    tuning_iter: int = Field(default=15, ge=1)
    enable_mlflow: bool = Field(default=True)


class PredictRequest(BaseModel):
    """Esquema de entrada para predecir un único registro desde la API.

    :param features: Diccionario con las variables necesarias para la inferencia.
    """
    features: dict[str, Any]
