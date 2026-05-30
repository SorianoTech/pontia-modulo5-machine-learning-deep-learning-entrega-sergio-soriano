import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "dataset_practica_final.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
METRICS_PATH = OUTPUTS_DIR / "metrics_summary.csv"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"file:{(BASE_DIR / 'mlruns').as_posix()}"
)
MLFLOW_EXPERIMENT_NAME = "hotel_cancellation_pipeline"
RUNS_DB_PATH = OUTPUTS_DIR / "runs.db"
LEGACY_RUNS_REGISTRY_PATH = OUTPUTS_DIR / "runs_registry.json"

TARGET_COLUMN = "is_canceled"
LEAKAGE_COLUMNS = ["reservation_status", "reservation_status_date", "company", "agent", "arrival_date_day_of_month", "days_in_waiting_list", "arrival_date_week_number"]
CHRONOLOGICAL_SPLIT_COLUMNS = ["arrival_date_year", "arrival_date_month"]


RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD_VALIDATION_SIZE = 0.2
DEFAULT_SPLIT_STRATEGY = "auto"

PRIMARY_METRIC = "roc_auc"
SECONDARY_METRICS = ["accuracy", "precision", "recall", "f1"]
DEFAULT_PREDICTION_THRESHOLD = 0.5
THRESHOLD_SELECTION_METRIC = "f1"
CALIBRATION_METHOD = "sigmoid"
CALIBRATION_CV = 3
PERMUTATION_IMPORTANCE_REPEATS = 5
EXPLAINABILITY_TOP_N = 20
DRIFT_TOP_FEATURES = 10
SERVING_CONTRACT_PATH = OUTPUTS_DIR / "serving_contract.json"
MONITORING_REPORT_PATH = OUTPUTS_DIR / "monitoring_report.json"
DEFAULT_TUNING_METHOD = "randomized"
DEFAULT_TUNING_CV = 3
DEFAULT_TUNING_ITER = 15

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
