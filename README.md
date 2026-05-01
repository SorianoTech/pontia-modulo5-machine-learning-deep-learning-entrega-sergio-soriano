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

## Bonus fase 2 implementados
- Tracking de experimentos con MLflow (runs, parametros, metricas y artefactos)
- Persistencia local de runs desde FastAPI en SQLite: `outputs/runs.db`
- Tuning de hiperparametros del mejor modelo base (Random Forest) con GridSearchCV o RandomizedSearchCV

## Tuning de hiperparametros

### Que es el tuning
El tuning (o ajuste de hiperparametros) es el proceso de buscar la combinacion optima de configuraciones de un modelo que maximiza su rendimiento. A diferencia de los parametros del modelo (que se aprenden durante el entrenamiento), los hiperparametros se fijan antes de entrenar y controlan el comportamiento del algoritmo. Por ejemplo, en Random Forest: el numero de arboles (`n_estimators`), la profundidad maxima (`max_depth`) o el numero de features a considerar en cada split (`max_features`).

En este proyecto el tuning se aplica sobre el modelo con mejor AUC-ROC base y busca mejorar su rendimiento sin cambiar el codigo de entrenamiento principal.

### GridSearchCV
Prueba **todas las combinaciones posibles** del espacio de busqueda definido. Es exhaustivo: si defines 3 valores para `n_estimators`, 3 para `max_depth` y 2 para `max_features`, evaluara 3 × 3 × 2 = 18 combinaciones, cada una con k-fold cross-validation.

**Ventaja:** garantiza encontrar el mejor resultado dentro del espacio definido.  
**Inconveniente:** coste computacional muy alto si el espacio de busqueda es grande.  
**Cuando usarlo:** espacios pequenos y cuando el tiempo de entrenamiento lo permite.

### RandomizedSearchCV
Prueba un numero fijo de combinaciones **elegidas al azar** del espacio de busqueda (controlado por `n_iter`). No evalua todas las posibilidades, sino una muestra representativa.

**Ventaja:** mucho mas rapido; con `n_iter=15-20` suele encontrar resultados muy proximos al optimo en una fraccion del tiempo.  
**Inconveniente:** no garantiza encontrar la combinacion exactamente optima.  
**Cuando usarlo:** espacios de busqueda grandes, recursos limitados o en fases de prototipado rapido.

### Comparativa rapida

| | GridSearchCV | RandomizedSearchCV |
|---|---|---|
| Tipo de busqueda | Exhaustiva | Aleatoria |
| Combinaciones evaluadas | Todas | `n_iter` (configurable) |
| Tiempo de ejecucion | Alto | Bajo |
| Garantia de optimo | Si (dentro del espacio) | No |
| Recomendado para | Espacios pequenos | Espacios grandes |

### Configuracion en este proyecto
El tuning se aplica sobre el pipeline completo (preprocesador + clasificador) del modelo ganador. Se puede activar desde la CLI o la API:

```bash
# Randomized (15 iteraciones, 3-fold CV)
python trainer.py --enable-tuning --tuning-method randomized --tuning-iter 15 --tuning-cv 3

# Grid (busqueda exhaustiva, 3-fold CV)
python trainer.py --enable-tuning --tuning-method grid --tuning-cv 3
```

El modelo tuneado se registra con el sufijo `_tuned` y compite con el resto en la tabla de metricas. Si supera al modelo base, pasa a ser el `best_model.pkl`.

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
python trainer.py --enable-tuning --tuning-method randomized --tuning-iter 20
python trainer.py --enable-tuning --tuning-method grid --tuning-cv 3
python trainer.py --disable-mlflow
```

## MLflow
El tracking URI se configura en modo local (file store) y guarda runs en `mlruns/`.

Para abrir UI de MLflow:
```bash
mlflow ui --backend-store-uri file:./mlruns
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

Endpoints bonus de persistencia de runs:
- `GET /runs`
- `GET /runs/{run_id}`

Endpoint extra de salud:
- `GET /health`

Ejemplo `POST /train`:
```json
{
  "quick_mode": true,
  "skip_neural_net": false,
  "enable_tuning": true,
  "tuning_method": "randomized",
  "tuning_cv": 3,
  "tuning_iter": 15,
  "enable_mlflow": true
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
