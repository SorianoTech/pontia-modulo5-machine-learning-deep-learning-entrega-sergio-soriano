from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from src import config
from src.model_trainer import ensure_probabilities


def load_best_model(path: str | None = None):
    """Carga el artefacto estructurado del mejor modelo.

    :param path: Ruta alternativa al fichero ``best_model.pkl``.
    :returns: Tupla con el nombre del modelo ganador y el objeto entrenado.
    """
    artifact_path = path if path else str(config.MODELS_DIR / "best_model.pkl")
    artifact = joblib.load(artifact_path)
    return artifact["model_name"], artifact["model"]


def predict_from_payload(payload: dict[str, Any], model_path: str | None = None) -> dict[str, Any]:
    """Ejecuta inferencia para un único registro a partir de un payload.

    :param payload: Diccionario con las variables de entrada del registro.
    :param model_path: Ruta alternativa al artefacto del mejor modelo.
    :returns: Diccionario con el nombre del modelo, la clase predicha y la
        probabilidad estimada de cancelación.
    """
    model_name, model = load_best_model(model_path)
    frame = pd.DataFrame([payload])
    # La función `ensure_probabilities` se encarga de verificar si el modelo tiene un método `predict_proba` para obtener las probabilidades de cada clase. Si el modelo no tiene este método, se asume que devuelve directamente las probabilidades en su método `predict`. En este caso, se obtiene la probabilidad de cancelación (la clase positiva) y se convierte a un valor flotante.
    proba = float(ensure_probabilities(model, frame)[0])
    # Si la probabilidad de cancelación es `>= 0.5`, devuelve `1`; si no, `0`. 
    # True o false se convierte a entero con `int()`, donde `True` se convierte a `1` y `False` a `0`.
    pred = int(proba >= 0.5)

    return {
        "model_name": model_name,
        "prediction": pred,
        "probability_canceled": round(proba, 6),
    }
