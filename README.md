# Proyecto Final ML - Cancelacion de Reservas Hoteleras

Este proyecto implementa un sistema automatico para entrenar, evaluar y comparar modelos de clasificacion binaria para predecir si una reserva hotelera sera cancelada (`is_canceled`).

## Autores

| Nombre              | Rol principal    |
|---------------------|------------------|
| Sergio Soriano      | Por definir      |
| Guillermo Parés     | Por definir      |

## Dataset
- Origen: `data/raw/dataset_practica_final.csv`
- Target: `is_canceled` (0 = no cancelada, 1 = cancelada)
- Columnas de leakage eliminadas: `reservation_status`, `reservation_status_date`

## Justificación del problema

Las cancelaciones de reservas hoteleras representan uno de los mayores retos operativos del sector. Cuando un cliente cancela, el hotel pierde ingresos que ya había planificado y, si la cancelación es tardía, difícilmente puede volver a vender esa habitación. En paralelo, el overbooking como estrategia defensiva genera experiencias negativas si no se calibra bien.

Predecir con antelación si una reserva va a cancelarse permite al hotel tomar decisiones proactivas:
- Ajustar la política de overbooking de forma inteligente.
- Priorizar esfuerzos comerciales en reservas de alto riesgo.
- Optimizar la gestión de inventario y precios dinámicos.

---

## Análisis exploratorio de datos

El EDA completo se encuentra en `notebooks/exploracion/eda_inicial.ipynb`. A continuación se recogen los hallazgos más relevantes.

### Desbalanceo de clases

El dataset presenta un desbalanceo moderado: aproximadamente el **37% de las reservas son canceladas** frente al 63% que no lo son.

### Valores nulos

| Columna | Nulos | Tratamiento |
|---|---|---|
| `children` | 4 | Imputación con 0 (ausencia de niños) |
| `country` | 488 | Imputación con `'Unknown'` |
| `agent` | 16.340 (13,7%) | Transformada en variable binaria `has_agent` |
| `company` | 112.593 (94,3%) | Eliminada — prácticamente vacía |

### Correlaciones con el target

Las variables con mayor correlación (positiva) con `is_canceled`:
- `lead_time`: a más antelación, más probabilidad de cancelación.
- `previous_cancellations`: historial de cancelaciones previas es el predictor más directo.
- `deposit_type` (No Refund): depósitos no reembolsables paradójicamente asociados a más cancelaciones.

Variables con correlación negativa (reducen probabilidad de cancelación):
- `total_of_special_requests`: más peticiones especiales indican mayor intención de viajar.
- `required_car_parking_spaces`: reservas con parking se cancelan menos.
- `booking_changes`: cambios en la reserva indican mayor compromiso del cliente.

---

## Modelos incluidos
- Regresion Logistica
- Arbol de Decision
- Random Forest
- Gradient Boosting (XGBoost si esta disponible, fallback a GradientBoostingClassifier)
- Red neuronal multicapa (MLP)

## Resultados

Entrenamiento completo sobre el dataset (119.390 reservas, split 80/20, semilla 42). Métrica principal: **AUC-ROC**.

| Modelo | Accuracy | Precision | Recall | F1-score | AUC-ROC |
|---|---|---|---|---|---|
| **Random Forest** | 0.8948 | 0.8940 | 0.8123 | 0.8512 | **0.9590** |
| Neural Network | 0.8728 | 0.8531 | 0.7931 | 0.8220 | 0.9447 |
| Gradient Boosting | 0.8626 | 0.8597 | 0.7518 | 0.8022 | 0.9397 |
| Decision Tree | 0.8296 | 0.7852 | 0.7434 | 0.7637 | 0.9092 |
| Logistic Regression | 0.8188 | 0.8119 | 0.6648 | 0.7310 | 0.8961 |

**Modelo ganador: Random Forest** con AUC-ROC de 0.959 — el mejor en todas las métricas. La red neuronal queda segunda en AUC-ROC pese a tener menos accuracy que Random Forest, lo que refleja mejor capacidad de separación entre clases.

---

## Bonus fase 2 implementados
- Tracking de experimentos con MLflow (runs, parametros, metricas, duraciones por etapa y artefactos) con backend PostgreSQL
- Servidor MLflow UI accesible en `http://localhost:5000`
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
├── .env.example
├── app.py
├── trainer.py
├── Dockerfile
├── docker-compose.yml
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
    ├── predictor.py
    ├── run_repository.py
    ├── tuning.py
    ├── mlflow_tracker.py
    ├── api_models.py
    └── api.py
```

## Arquitectura general
```
dataset_practica_final.csv
        │
        ▼
  data_loader.py          → carga, limpieza, split 80/20, preprocesador
        │
        ▼
  model_trainer.py        → entrena 5 modelos en pipeline (preprocesador + clasificador)
        │
        ▼
  evaluator.py            → calcula métricas, genera gráficos ROC / CM / feature importance
        │
        ▼
  tuning.py (opcional)    → GridSearchCV / RandomizedSearchCV sobre el mejor modelo
        │
        ▼
  best_model.pkl          → modelo ganador serializado con joblib
        │
        ▼
  mlflow_tracker.py       → registra params, métricas y artefactos en MLflow
```

## Instalacion
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

## Docker Compose
Se incluye una configuracion lista para levantar todos los servicios como contenedores independientes.

### Servicios
| Servicio | Responsabilidad | Puerto |
|---|---|---|
| `api` | FastAPI — inferencia y endpoints REST | 8000 |
| `streamlit` | Interfaz de usuario | 8501 |
| `mlflow` | Servidor MLflow UI y tracking | 5000 |
| `postgres` | Backend de almacenamiento de MLflow | 5432 |
| `trainer` | Job de entrenamiento batch (perfil `jobs`) | — |

### Archivos incluidos
- `Dockerfile`: imagen Python 3.11 multi-stage compartida por todos los servicios de la aplicacion.
- `docker-compose.yml`: orquestacion completa con dependencias y healthchecks.
- `.env.example`: plantilla de variables de entorno.

### Puesta en marcha
```bash
# Windows PowerShell:
Copy-Item .env.example .env

# Bash:
cp .env.example .env

# Edita .env y cambia POSTGRES_PASSWORD antes de continuar

docker compose up --build
```

Servicios expuestos:
- API FastAPI: `http://localhost:8000`
- Healthcheck API: `http://localhost:8000/health`
- Streamlit: `http://localhost:8501`
- MLflow UI: `http://localhost:5000`

### Orden de arranque
```
postgres (healthy)
    └── mlflow
            ├── api (healthy)
            │       └── streamlit
            └── trainer  (solo con --profile jobs)
```

### Entrenar desde Docker
Opcion 1: usando la interfaz Streamlit (`http://localhost:8501`), que delega el entrenamiento en la API.

Opcion 2: lanzando el job de entrenamiento:
```bash
docker compose run --rm trainer python trainer.py --quick --skip-neural-net
```

Los artefactos persistentes se guardan en volumenes Docker:
- `models` -> modelos serializados (`.pkl`)
- `outputs` -> metricas CSV, graficos PNG y `runs.db`
- `mlruns` -> artefactos de MLflow
- `postgres_data` -> base de datos PostgreSQL de MLflow

### Variables de entorno (`.env`)
| Variable | Descripcion | Default |
|---|---|---|
| `APP_IMAGE` | Tag de la imagen Docker | `hotel-cancellation-ml:latest` |
| `API_PORT` | Puerto host para la API | `8000` |
| `STREAMLIT_PORT` | Puerto host para Streamlit | `8501` |
| `MLFLOW_PORT` | Puerto host para MLflow UI | `5000` |
| `UVICORN_LOG_LEVEL` | Nivel de log de uvicorn | `info` |
| `POSTGRES_USER` | Usuario de PostgreSQL | `mlflow` |
| `POSTGRES_PASSWORD` | Contrasena de PostgreSQL | `mlflow` |
| `POSTGRES_DB` | Base de datos de PostgreSQL | `mlflow` |
| `POSTGRES_PORT` | Puerto host para PostgreSQL | `5432` |

### Notas
- La primera vez, `predict` y la vista de evaluacion no tendran artefactos hasta ejecutar un entrenamiento.
- El dataset se monta como volumen de solo lectura desde `./data` — no es necesario reconstruir la imagen para cambiar los datos.
- Cambia `POSTGRES_PASSWORD` en `.env` antes de desplegar en produccion.

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
MLflow corre como servicio independiente con backend PostgreSQL y almacenamiento de artefactos en el volumen `mlruns`.

Con Docker Compose, la UI esta disponible automaticamente en `http://localhost:5000` al hacer `docker compose up`.

Para desarrollo local sin Docker:
```bash
# Usar file store local
mlflow ui --backend-store-uri file:./mlruns

# O apuntar al servidor de PostgreSQL si esta corriendo
export MLFLOW_TRACKING_URI=http://localhost:5000
python trainer.py
```

El tracking URI puede sobreescribirse con la variable de entorno `MLFLOW_TRACKING_URI`. Todos los contenedores la reciben configurada automaticamente via `docker-compose.yml`.

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
