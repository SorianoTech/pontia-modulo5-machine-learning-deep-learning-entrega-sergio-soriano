# Explicacion de `Dockerfile` y `docker-compose.yml`

Este documento resume la estructura Docker del proyecto despues de separar desarrollo y produccion.

## Dockerfile

El `Dockerfile` esta dividido en targets para no reutilizar la misma imagen pesada en todos los casos:

- `builder-app`: instala las dependencias necesarias para ejecutar la aplicacion.
- `builder-dev`: anade utilidades de desarrollo sobre las dependencias de la aplicacion.
- `builder-mlflow`: instala solo lo necesario para MLflow.
- `app-runtime`: imagen de produccion para API, Streamlit y trainer.
- `dev`: imagen de desarrollo para API, Streamlit y trainer.
- `mlflow-runtime`: imagen de produccion dedicada a MLflow.

### Base comun

`python-base` fija:

- la imagen `python:3.11.9-slim`,
- variables de entorno de Python y `pip`,
- `WORKDIR /app`.

`builder-base` instala `build-essential`, crea `/opt/venv` y deja el `PATH` apuntando al entorno virtual.

### Requirements separados

Cada builder usa un archivo distinto:

- `requirements-app.txt`: dependencias de ejecucion de la aplicacion.
- `requirements-dev.txt`: suma `pytest` e `ipykernel`.
- `requirements-mlflow.txt`: mantiene MLflow separado del stack completo de ML.

Esto mejora cacheo y reduce el peso de las imagenes de produccion.

### `app-runtime`

`app-runtime`:

- instala solo librerias de sistema necesarias en runtime,
- crea el usuario no privilegiado `appuser`,
- copia el entorno virtual desde `builder-app`,
- copia `app.py`, `trainer.py`, `README.md` y `src/`,
- prepara `models/`, `outputs/`, `mlruns/` y `data/`.

El directorio `data/` ya no se copia dentro de la imagen: se monta como volumen para evitar reconstrucciones al cambiar el dataset.

### `dev`

`dev` reutiliza `app-runtime` y reemplaza el entorno virtual por el construido en `builder-dev`.

Asi el entorno de desarrollo mantiene utilidades extra sin meterlas en produccion.

### `mlflow-runtime`

`mlflow-runtime` es una imagen separada y mas ligera:

- instala `libpq5`,
- copia el entorno virtual de `builder-mlflow`,
- prepara `/app/mlruns`,
- expone el puerto `5000`.

Esto evita que MLflow cargue TensorFlow, CatBoost, Streamlit y el resto de dependencias de la aplicacion principal.

## `docker-compose.yml`

El archivo base queda orientado a produccion:

- usa `image:` en lugar de `build:`,
- espera imagenes preconstruidas (`APP_IMAGE` y `MLFLOW_IMAGE`),
- conserva healthchecks, volumenes y dependencias entre servicios.

### Servicios

- `api`, `streamlit` y `trainer` comparten `APP_IMAGE`.
- `mlflow` usa `MLFLOW_IMAGE`.
- `postgres` sigue ejecutandose sobre `postgres:16-alpine`.

El dataset se monta como `./data:/app/data:ro`, asi que cambiar datos no invalida el build.

## `docker-compose.override.yml`

El override queda centrado en desarrollo local:

- anade `build:` con target `dev` para `api`, `streamlit` y `trainer`,
- anade `build:` con target `mlflow-runtime` para `mlflow`,
- monta `src/`, `app.py` y `trainer.py`,
- activa autorecarga en FastAPI y Streamlit.

Docker Compose carga este archivo automaticamente en desarrollo; en produccion no deberia usarse.

## Flujo recomendado

- Desarrollo: `docker compose up --build`
- Produccion: construir o publicar las imagenes y lanzar `docker compose -f docker-compose.yml up -d`

## Beneficios

- Produccion ya no incluye dependencias de desarrollo.
- MLflow queda aislado en una imagen propia.
- El dataset deja de hornearse dentro de la imagen.
- El compose base sirve mejor para despliegues reproducibles.
