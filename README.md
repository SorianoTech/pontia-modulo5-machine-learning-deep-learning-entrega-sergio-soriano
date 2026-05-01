# Proyecto Final ML - Cancelacion de Reservas Hoteleras

Este proyecto implementa un sistema automatico para entrenar, evaluar y comparar modelos de clasificacion binaria para predecir si una reserva hotelera sera cancelada (`is_canceled`).

## Dataset
- Origen: `data/raw/dataset_practica_final.csv`
- Target: `is_canceled` (0 = no cancelada, 1 = cancelada)
- Columnas de leakage eliminadas: `reservation_status`, `reservation_status_date`

## Modelos incluidos
- Regresion Logistica
- Arbol de Decision
- Random Forest
- Gradient Boosting (XGBoost si esta disponible, fallback a GradientBoostingClassifier)
- Red neuronal multicapa (MLP)

## Justificacion de metricas
**Metrica principal: AUC-ROC**

Se utiliza AUC-ROC como metrica principal porque:
1. El problema de cancelacion de reservas suele tener desbalance de clases.
2. AUC-ROC mide la capacidad de separacion entre clases para todos los umbrales.
3. Es menos sensible a desbalance que accuracy como metrica unica.

**Metricas secundarias**
- Accuracy: referencia global, util pero puede ser enganosa con desbalance.
- Precision: controla falsos positivos (predecir cancelacion cuando no ocurre).
- Recall: controla falsos negativos (no detectar cancelaciones reales).
- F1-score: equilibrio entre precision y recall.

## Estructura
```
entregable/
├── .github/workflows/ci.yml
├── app.py
├── trainer.py
├── requirements.txt
├── data/raw/dataset_practica_final.csv
├── notebooks/
│   ├── exploracion/eda_inicial.ipynb
│   └── finales/comparativa_modelos.ipynb
├── models/
├── outputs/
└── src/
    ├── config.py
    ├── data_loader.py
    ├── model_trainer.py
    ├── evaluator.py
    ├── predictor.py
    └── api.py
```

## Instalacion
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecucion del pipeline
```bash
python trainer.py
```
Opciones:
```bash
python trainer.py --quick
python trainer.py --quick --skip-neural-net
```

## API FastAPI
Lanzar servidor:
```bash
uvicorn src.api:app --reload
```

Endpoints requeridos:
- `POST /train`
- `POST /predict`
- `GET /evaluate`

Endpoint extra de salud:
- `GET /health`

Ejemplo `POST /train`:
```json
{
  "quick_mode": true,
  "skip_neural_net": false
}
```

Ejemplo `POST /predict`:
```json
{
  "features": {
    "hotel": "Resort Hotel",
    "lead_time": 20,
    "arrival_date_year": 2016,
    "arrival_date_month": "July",
    "arrival_date_week_number": 27,
    "arrival_date_day_of_month": 1,
    "stays_in_weekend_nights": 1,
    "stays_in_week_nights": 2,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "meal": "BB",
    "country": "PRT",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A",
    "assigned_room_type": "A",
    "booking_changes": 0,
    "deposit_type": "No Deposit",
    "agent": 240,
    "company": null,
    "days_in_waiting_list": 0,
    "customer_type": "Transient",
    "adr": 100.0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 1
  }
}
```

## Streamlit
Lanzar interfaz:
```bash
streamlit run app.py
```

La app incluye:
- Entrenar modelos
- Evaluar resultados y graficos
- Predecir una reserva individual

## CI/CD
Workflow en GitHub Actions:
- Instala dependencias
- Ejecuta entrenamiento smoke test
- Verifica que la API responde en `/health`
