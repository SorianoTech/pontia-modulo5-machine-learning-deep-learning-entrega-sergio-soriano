FROM python:3.11.9-slim AS python-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

FROM python-base AS builder-base

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

FROM builder-base AS builder-app

COPY requirements-app.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements-app.txt

FROM builder-base AS builder-dev

COPY requirements-app.txt requirements-dev.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements-dev.txt

FROM builder-base AS builder-mlflow

COPY requirements-mlflow.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements-mlflow.txt

FROM python-base AS app-runtime

ENV GIT_PYTHON_REFRESH=quiet \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libfreetype6 libpng16-16 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system appuser \
    && useradd --system --gid appuser --create-home --home-dir /home/appuser appuser

COPY --from=builder-app /opt/venv /opt/venv
COPY --chown=appuser:appuser app.py trainer.py README.md ./
COPY --chown=appuser:appuser src ./src

RUN mkdir -p /app/models /app/outputs /app/mlruns /app/data \
    && chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 8000 8501

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM app-runtime AS dev

COPY --from=builder-dev /opt/venv /opt/venv

FROM python-base AS mlflow-runtime

ENV GIT_PYTHON_REFRESH=quiet \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system appuser \
    && useradd --system --gid appuser --create-home --home-dir /home/appuser appuser

COPY --from=builder-mlflow /opt/venv /opt/venv

RUN mkdir -p /app/mlruns \
    && chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
