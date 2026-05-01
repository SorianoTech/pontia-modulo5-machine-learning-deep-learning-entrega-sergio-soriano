from __future__ import annotations

import ast

import pandas as pd
import streamlit as st

from src import config
from src.data_loader import build_data_bundle
from src.predictor import predict_from_payload
from trainer import run_pipeline

st.set_page_config(page_title="Hotel Cancellation ML", layout="wide")

st.title("Proyecto Final ML - Hotel Booking Cancellation")
st.caption("Entrenamiento, evaluación e inferencia con interfaz Streamlit")

mode = st.sidebar.radio("Selecciona una sección", ["Entrenar", "Evaluar", "Predecir"])

if mode == "Entrenar":
    st.subheader("Entrenamiento")
    quick_mode = st.checkbox("Quick mode", value=True)
    skip_neural_net = st.checkbox("Saltar red neuronal", value=False)

    if st.button("Ejecutar entrenamiento", type="primary"):
        with st.spinner("Entrenando modelos..."):
            result = run_pipeline(quick_mode=quick_mode, skip_neural_net=skip_neural_net)

        st.success(f"Mejor modelo: {result['best_model']}")
        st.dataframe(result["metrics"], use_container_width=True)

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

elif mode == "Predecir":
    st.subheader("Predicción individual")

    try:
        bundle = build_data_bundle()
        sample = bundle.X_test.iloc[0].to_dict()
    except Exception:
        sample = {}

    st.write("Rellena los campos en formato JSON. Puedes usar este ejemplo:")
    st.code(str(sample), language="python")

    input_text = st.text_area("Features (dict)", value=str(sample), height=220)

    if st.button("Predecir", type="primary"):
        try:
            payload = ast.literal_eval(input_text)
            result = predict_from_payload(payload)
            st.success(f"Predicción: {result['prediction']} | Prob cancelación: {result['probability_canceled']}")
            st.json(result)
        except FileNotFoundError:
            st.error("No existe best_model.pkl. Entrena primero un modelo.")
        except Exception as exc:
            st.error(f"Error en la predicción: {exc}")
