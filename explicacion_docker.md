# Explicación de `Dockerfile` y `docker-compose.yml`

Este documento describe **por qué existe cada paso** de los dos archivos de contenedorización del proyecto y qué aporta dentro de la solución.

## Dockerfile

El `Dockerfile` está construido en **dos etapas** (`builder` y `runtime`) para separar la instalación de dependencias de compilación del contenedor final de ejecución.

### Etapa 1: `builder`

#### `FROM python:3.11.9-slim AS builder`
Se parte de una imagen oficial de Python, ligera (`slim`), para reducir tamaño. Se nombra como `builder` porque esta etapa se usa para **preparar dependencias**.

#### `ENV PYTHONDONTWRITEBYTECODE=1 ...`
Estas variables ajustan el comportamiento de Python y `pip`:

- `PYTHONDONTWRITEBYTECODE=1`: evita crear archivos `.pyc`, reduciendo ruido y escrituras innecesarias.
- `PYTHONUNBUFFERED=1`: fuerza salida inmediata por consola, útil para ver logs en Docker sin retraso.
- `PIP_NO_CACHE_DIR=1`: evita que `pip` guarde caché y así reduce tamaño de imagen.
- `PIP_DISABLE_PIP_VERSION_CHECK=1`: elimina comprobaciones de versión de `pip`, acelerando un poco la instalación.

#### `WORKDIR /app`
Define `/app` como directorio de trabajo. Así, los siguientes `COPY`, `RUN` y el arranque se hacen desde una ruta consistente.

#### `RUN apt-get update && apt-get install ... build-essential`
Instala herramientas de compilación (`build-essential`) porque algunas dependencias de Python pueden necesitar compilar extensiones nativas durante `pip install`.

El borrado de `/var/lib/apt/lists/*` al final:

- reduce el tamaño de la capa,
- evita dejar metadatos de paquetes ya innecesarios.

#### `RUN python -m venv /opt/venv`
Crea un entorno virtual dentro del contenedor. Esto aísla las dependencias del sistema y facilita copiar todo el entorno ya preparado a la imagen final.

#### `ENV PATH="/opt/venv/bin:${PATH}"`
Hace que el contenedor use por defecto el Python y los binarios instalados dentro del entorno virtual.

#### `COPY requirements.txt .`
Se copia primero solo `requirements.txt` para aprovechar la caché de Docker: si cambia el código pero no las dependencias, la instalación no se repite.

#### `RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt`
Primero actualiza herramientas básicas de empaquetado para evitar problemas de instalación. Después instala todas las dependencias del proyecto en el entorno virtual.

### Etapa 2: `runtime`

#### `FROM python:3.11.9-slim AS runtime`
Se vuelve a partir de una imagen ligera limpia para el contenedor final. Así no se arrastran herramientas de compilación innecesarias de la etapa `builder`.

#### `ENV ... GIT_PYTHON_REFRESH=quiet ...`
Se repiten variables útiles de Python y `pip`, y además:

- `GIT_PYTHON_REFRESH=quiet`: evita mensajes innecesarios de librerías que inspeccionan Git.
- `PATH="/opt/venv/bin:${PATH}"`: permite usar directamente las dependencias copiadas desde `/opt/venv`.

#### `WORKDIR /app`
Mantiene el mismo directorio de trabajo que en la etapa anterior para que la estructura sea coherente.

#### `RUN apt-get update && apt-get install ... libgomp1 libfreetype6 libpng16-16 ...`
Instala librerías de sistema necesarias en tiempo de ejecución:

- `libgomp1`: habitual en librerías de machine learning que usan paralelización/OpenMP.
- `libfreetype6` y `libpng16-16`: comunes para generación de gráficos o renderizado de imágenes.

En el mismo bloque también se crea un usuario no privilegiado:

- `groupadd --system appuser`
- `useradd --system ... appuser`

Esto mejora la seguridad, porque la aplicación no se ejecuta como `root`.

#### `COPY --from=builder /opt/venv /opt/venv`
Copia el entorno virtual ya montado en la etapa `builder`. Así se reutiliza todo lo instalado sin recompilar ni reinstalar.

#### `COPY --chown=appuser:appuser app.py trainer.py requirements.txt README.md ./`
#### `COPY --chown=appuser:appuser src ./src`
#### `COPY --chown=appuser:appuser data ./data`
Se copian los archivos de la aplicación y se asigna su propiedad a `appuser`. El uso de `--chown` evita problemas de permisos cuando el contenedor deja de ejecutarse como `root`.

#### `RUN mkdir -p /app/models /app/outputs /app/mlruns /app/data && chown -R appuser:appuser /app /home/appuser`
Prepara directorios donde la aplicación va a escribir o leer:

- `/app/models`: modelos entrenados.
- `/app/outputs`: salidas generadas.
- `/app/mlruns`: artefactos y seguimiento de MLflow.
- `/app/data`: datos disponibles dentro del contenedor.

Después ajusta permisos para que el usuario de aplicación pueda trabajar sobre ellos.

#### `USER appuser`
Desde este punto el contenedor se ejecuta con el usuario no privilegiado. Es una práctica de seguridad recomendada.

#### `EXPOSE 8000 8501`
Documenta que la imagen está pensada para usar:

- `8000` para la API con Uvicorn,
- `8501` para Streamlit.

No publica puertos por sí mismo, pero deja claro qué servicios expone la imagen.

#### `CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]`
Define el comando por defecto de la imagen: arrancar la API FastAPI/Uvicorn escuchando en todas las interfaces del contenedor. Esto permite que Docker Compose o el host accedan al servicio.

---

## docker-compose.yml

`docker-compose.yml` orquesta varios servicios que comparten la misma imagen base del proyecto, cambiando únicamente el comando de arranque según el rol.

### `services:`
Agrupa todos los contenedores que forman la solución completa.

## Servicio `api`

### `image: ${APP_IMAGE:-hotel-cancellation-ml:latest}`
Permite reutilizar un nombre de imagen configurable por variable de entorno. Si no se define, usa `hotel-cancellation-ml:latest`.

### `build:`
Indica que la imagen se construye localmente:

- `context: .`: usa la carpeta actual como contexto de build.
- `dockerfile: Dockerfile`: usa ese archivo como receta de construcción.

### `command: ... uvicorn ...`
Sobrescribe el `CMD` del `Dockerfile` de forma explícita dentro de Compose. Esto hace visible cómo arranca la API y permite parametrizar el nivel de log.

### `ports:`
Mapea el puerto del host al puerto interno `8000`. La variable `API_PORT` permite cambiar el puerto externo sin editar el YAML.

### `environment:`

- `PYTHONUNBUFFERED: "1"`: logs inmediatos.
- `MLFLOW_TRACKING_URI: "http://mlflow:5000"`: la API sabe que MLflow vive en el servicio `mlflow` dentro de la red interna de Compose.

### `volumes:`
Persisten artefactos aunque se recree el contenedor:

- `models`: modelos entrenados.
- `outputs`: resultados generados.
- `mlruns`: experimentos de MLflow.

### `restart: unless-stopped`
Hace que el contenedor se reinicie automáticamente salvo que se haya detenido manualmente.

### `healthcheck:`
Comprueba que la API responde en `/health`. No basta con que el proceso exista: se valida que el servicio está operativo.

### `depends_on:`
Declara dependencia con `mlflow` para que la API no arranque antes de que ese servicio esté al menos iniciado.

## Servicio `streamlit`

Usa la misma imagen, pero cambia el comando para lanzar la interfaz Streamlit.

### `command: ... streamlit run app.py ...`
Arranca la aplicación web y fuerza:

- escucha en `0.0.0.0` para acceso externo,
- puerto `8501`,
- modo `headless`, adecuado para contenedores.

### `ports:`
Publica `8501` en el host, configurable con `STREAMLIT_PORT`.

### `environment:`

- `API_URL: "http://api:8000"`: Streamlit consume la API usando el nombre del servicio dentro de la red de Compose.
- `MLFLOW_TRACKING_URI: "http://mlflow:5000"`: acceso al servidor de tracking.

### `volumes:`
Comparte modelos, salidas y experimentos con otros servicios. Además:

- `./data:/app/data:ro`: monta los datos del proyecto en modo solo lectura para evitar modificaciones accidentales desde el contenedor.

### `healthcheck:`
Comprueba que la interfaz web responde antes de considerarla sana.

### `depends_on:`

- espera a que `api` esté **healthy**, no solo arrancada;
- espera a que `mlflow` esté iniciado.

Esto evita que la interfaz intente usar servicios aún no disponibles.

## Servicio `trainer`

### `command: ["python", "trainer.py"]`
Ejecuta el script de entrenamiento de modelos.

### `profiles: ["jobs"]`
Evita que se lance siempre. Solo se ejecuta si se activa el perfil `jobs`, útil porque entrenar es una tarea puntual y no un servicio permanente.

### `restart: "no"`
No debe reiniciarse automáticamente: si el entrenamiento termina, el contenedor debe finalizar.

### `volumes` y `environment`
Comparte modelos, salidas, `mlruns` y acceso de solo lectura a datos para que el entrenamiento deje resultados persistentes y registrables en MLflow.

## Servicio `mlflow`

### `command: ... mlflow server ...`
Levanta el servidor de MLflow.

Los parámetros indican:

- `--host 0.0.0.0`: accesible desde otros contenedores y desde el host si se publica el puerto.
- `--port 5000`: puerto estándar elegido para el servicio.
- `--backend-store-uri postgresql://...`: usa PostgreSQL para guardar metadatos, experimentos y ejecuciones.
- `--default-artifact-root file:///app/mlruns`: guarda artefactos en el volumen compartido `mlruns`.

### `ports:`
Publica el puerto de MLflow y permite cambiarlo con `MLFLOW_PORT`.

### `volumes:`
`mlruns` persiste los artefactos aunque el contenedor se recree.

### `environment:`
Centraliza credenciales y nombre de base de datos usando variables de entorno con valores por defecto.

### `depends_on:`
Espera a que `postgres` esté saludable antes de arrancar, porque MLflow necesita la base de datos disponible.

## Servicio `postgres`

### `image: postgres:16-alpine`
Usa PostgreSQL oficial en variante `alpine`, más ligera.

### `environment:`
Configura usuario, contraseña y base de datos inicial para MLflow.

### `#ports:`
Está comentado, lo que sugiere que PostgreSQL no necesita exponerse al host normalmente. Se mantiene solo accesible dentro de la red interna de Compose, mejorando seguridad.

### `volumes:`
`postgres_data` persiste los datos de la base para no perderlos al recrear el contenedor.

### `healthcheck:`
Usa `pg_isready` para comprobar que PostgreSQL acepta conexiones antes de que otros servicios dependientes intenten usarla.

## `volumes:`

Se declaran volúmenes nombrados para que Docker gestione almacenamiento persistente de:

- `models`
- `outputs`
- `mlruns`
- `postgres_data`

La ventaja es que los datos sobreviven a reinicios y recreaciones de contenedores, y además pueden compartirse entre servicios cuando corresponde.
