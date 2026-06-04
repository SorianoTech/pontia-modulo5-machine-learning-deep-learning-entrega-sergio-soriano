from __future__ import annotations

import ast
import json
import os

import httpx
import pandas as pd
import streamlit as st

from src import config
from src.data_loader import build_data_bundle
from src.predictor import predict_from_payload
from src.run_repository import list_runs

API_URL = os.getenv("API_URL", "http://api:8000")


def _sanitize_for_json(value):
    """Normaliza valores anidados para que puedan serializarse como JSON.

    :param value: Valor escalar o estructura anidada que puede contener objetos
        de pandas.
    :returns: Estructura equivalente formada solo por tipos compatibles con JSON.
    """
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if pd.isna(value):
        return None
    return value


def _parse_payload(text: str) -> dict:
    """Interpreta el texto introducido en la UI como JSON o literal de Python.

    :param text: Contenido pegado por el usuario en el cuadro de texto.
    :returns: Diccionario listo para enviarse al flujo de predicción.
    :raises ValueError: Si el contenido no representa un diccionario.
    """
    # Prefer strict JSON parsing first.
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Fallback for Python-style dicts pasted by the user.
        payload = ast.literal_eval(text)

    if not isinstance(payload, dict):
        raise ValueError("El payload debe ser un objeto JSON (diccionario clave-valor).")
    return payload

st.set_page_config(page_title="Hotel Cancellation ML", layout="wide")

st.title("Proyecto Final ML - Hotel Booking Cancellation")
st.caption("Entrenamiento, evaluación e inferencia con interfaz Streamlit")

mode = st.sidebar.radio("Selecciona una sección", ["Entrenar", "Evaluar", "Predecir"])

if mode == "Entrenar":
    st.subheader("Entrenamiento")
    quick_mode = st.checkbox("Quick mode", value=True)
    skip_neural_net = st.checkbox("Saltar red neuronal", value=False)
    enable_tuning = st.checkbox("Activar tuning (Grid/Randomized)", value=False)
    tuning_method = st.selectbox("Método de tuning", options=["randomized", "grid"], index=0)
    tuning_cv = st.slider("Folds CV", min_value=2, max_value=5, value=3)
    tuning_iter = st.slider("Iteraciones Randomized", min_value=5, max_value=40, value=15)
    enable_mlflow = st.checkbox("Activar tracking MLflow", value=True)

    if st.button("Ejecutar entrenamiento", type="primary"):
        with st.spinner("Entrenando modelos..."):
            try:
                resp = httpx.post(
                    f"{API_URL}/train",
                    json={
                        "quick_mode": quick_mode,
                        "skip_neural_net": skip_neural_net,
                        "enable_tuning": enable_tuning,
                        "tuning_method": tuning_method,
                        "tuning_cv": tuning_cv,
                        "tuning_iter": tuning_iter,
                        "enable_mlflow": enable_mlflow,
                    },
                    timeout=600.0,
                )
                resp.raise_for_status()
                result = resp.json()
            except httpx.RequestError as exc:
                st.error(f"Error al conectar con la API: {exc}")
                st.stop()
            except httpx.HTTPStatusError as exc:
                st.error(f"Error en la API ({exc.response.status_code}): {exc.response.text}")
                st.stop()

        st.success(f"Mejor modelo: {result['best_model']}")
        if result.get("mlflow_run_id"):
            st.info(f"MLflow run id: {result['mlflow_run_id']}")
        if result.get("tuning"):
            st.json({"tuning": result["tuning"]})
        st.dataframe(pd.DataFrame(result["metrics"]), use_container_width=True)

elif mode == "Evaluar":
    st.subheader("Evaluación")
    metrics_path = config.METRICS_PATH
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.info("No hay métricas. Ejecuta primero la sección Entrenar.")

    roc_path = config.OUTPUTS_DIR / "roc_curves.png"
    if roc_path.exists():
        st.image(str(roc_path), caption="Curvas ROC")

    fi_path = config.OUTPUTS_DIR / "feature_importance_random_forest.png"
    if fi_path.exists():
        st.image(str(fi_path), caption="Importancia de variables (Random Forest)")

    st.divider()
    st.subheader("Histórico de runs")
    runs = list_runs(limit=20)
    if runs:
        st.dataframe(pd.DataFrame(runs), use_container_width=True)
    else:
        st.info("No hay runs persistidos todavía.")

elif mode == "Predecir":
    st.subheader("Predicción individual")

    try:
        bundle = build_data_bundle()
        sample = _sanitize_for_json(bundle.X_test.iloc[0].to_dict())
    except Exception:
        sample = {}

    example_text = json.dumps(sample, ensure_ascii=False, indent=2)

    st.write("Rellena los campos en formato JSON. Puedes usar este ejemplo:")
    st.code(example_text, language="json")

    input_text = st.text_area("Features (JSON)", value=example_text, height=260)

    if st.button("Predecir", type="primary"):
        try:
            payload = _parse_payload(input_text)
            result = predict_from_payload(payload)
            st.success(f"Predicción: {result['prediction']} | Prob cancelación: {result['probability_canceled']}")
            st.json(result)
        except FileNotFoundError:
            st.error("No existe best_model.pkl. Entrena primero un modelo.")
        except Exception as exc:
            st.error(f"Error en la predicción: {exc}")
