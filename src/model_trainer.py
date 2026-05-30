from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False

from src import config
from src.data_loader import DataBundle, split_data

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class KerasClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, epochs: int = 50, batch_size: int = 256, learning_rate: float = 0.001) -> None:

        # Número de veces que el modelo verá todos los datos de entrenamiento
        self.epochs = epochs

        # Número de muestras que se procesan juntas antes de actualizar los pesos
        self.batch_size = batch_size

        # Velocidad a la que el optimizador ajusta los pesos
        self.learning_rate = learning_rate

    def fit(self, X, y):

        # Keras trabaja con float32; el preprocesador de sklearn devuelve float64
        X = np.array(X, dtype="float32")
        y = np.array(y, dtype="float32")

        # Arquitectura de la red: tres capas densas (fully connected)
        #   - Capa 1: 64 neuronas con ReLU
        #   - Capa 2: 32 neuronas con ReLU
        #   - Capa 3: 1 neurona con sigmoide → devuelve una probabilidad entre 0 y 1
        self.model_ = keras.Sequential([
            keras.layers.Dense(64, activation="relu", input_shape=(X.shape[1],)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ])

        # Configuración del entrenamiento:
        #   - Adam: optimizador que adapta el learning rate automáticamente
        #   - binary_crossentropy: función de pérdida estándar para clasificación binaria
        self.model_.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
        )

        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.array(X, dtype="float32")

        # model_.predict devuelve shape (n, 1) → flatten lo convierte a (n,)
        proba = self.model_.predict(X, verbose=0).flatten()

        # sklearn espera columnas [prob_clase_0, prob_clase_1]
        return np.column_stack([1 - proba, proba])

    def predict(self, X) -> np.ndarray:
        
        # Si la probabilidad de cancelación supera 0.5 → predice 1 (cancelado)
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class CatBoostClassifierWrapper(BaseEstimator, ClassifierMixin):

    def __init__(self, iterations: int = 100, learning_rate: float = 0.1, depth: int = 6) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth

    def _prepare_features(self, X):
        if not hasattr(X, "copy"):
            return X

        X_prepared = X.copy()
        if not hasattr(X_prepared, "select_dtypes"):
            return X_prepared

        cat_cols = X_prepared.select_dtypes(exclude=["number"]).columns.tolist()
        for col in cat_cols:
            X_prepared[col] = X_prepared[col].fillna("Unknown").astype(str)
        return X_prepared

    def fit(self, X, y):
        X_prepared = self._prepare_features(X)
        cat_features = None
        if hasattr(X_prepared, "select_dtypes"):
            cat_features = X_prepared.select_dtypes(exclude=["number"]).columns.tolist()

        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            verbose=0,
        )
        self.model_.fit(X_prepared, y, cat_features=cat_features)
        return self

    def evaluate(self, X_test, y_test) -> None:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        conf = confusion_matrix(y_test, y_pred)
        clf_report = classification_report(y_test, y_pred)

        print(f"Accuracy Score of CatBoost Classifier is : {acc}")
        print(f"Confusion Matrix : \n{conf}")
        print(f"Classification Report : \n{clf_report}")

    def predict_proba(self, X) -> np.ndarray:
        X_prepared = self._prepare_features(X)
        return self.model_.predict_proba(X_prepared)

    def predict(self, X) -> np.ndarray:
        X_prepared = self._prepare_features(X)
        return self.model_.predict(X_prepared).flatten()


@dataclass
class TrainedModels:
    models: Dict[str, BaseEstimator]
    model_paths: Dict[str, Path]


class ModelTrainer:
    def __init__(self, data_bundle: DataBundle, quick_mode: bool = False, skip_neural_net: bool = False) -> None:
        self.data = data_bundle
        self.quick_mode = quick_mode
        self.skip_neural_net = skip_neural_net

    def _get_estimators(self) -> Dict[str, ClassifierMixin]:
        estimators: Dict[str, ClassifierMixin] = {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
            "decision_tree": DecisionTreeClassifier(max_depth=8, random_state=config.RANDOM_STATE),
            "random_forest": RandomForestClassifier(
                n_estimators=50 if self.quick_mode else 200,
                max_depth=None,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            ),
            "gradient_boosting": self._build_gradient_boosting(),
        }

        if HAS_CATBOOST:
            estimators["catboost"] = CatBoostClassifierWrapper(
                iterations=50 if self.quick_mode else 100,
            )

        if not self.skip_neural_net:
            if not HAS_KERAS:
                raise ImportError("Instala TensorFlow con: pip install tensorflow")
            estimators["neural_network"] = KerasClassifier(
                epochs=10 if self.quick_mode else 50,
            )

        return estimators

    def _build_gradient_boosting(self) -> ClassifierMixin:
        if HAS_XGBOOST:
            return XGBClassifier(
                n_estimators=40 if self.quick_mode else 200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            )

        # Fallback when XGBoost is not available.
        return GradientBoostingClassifier(random_state=config.RANDOM_STATE)

    def train_all(self) -> TrainedModels:
        estimators = self._get_estimators()
        trained: Dict[str, BaseEstimator] = {}
        paths: Dict[str, Path] = {}

        for name, estimator in estimators.items():
            if name == "catboost":
                model = estimator
            else:
                model = Pipeline(
                    steps=[
                        ("preprocessor", self.data.preprocessor),
                        ("classifier", estimator),
                    ]
                )
            model.fit(self.data.X_train, self.data.y_train)
            trained[name] = model

            model_path = config.MODELS_DIR / f"{name}.pkl"
            joblib.dump(model, model_path)
            paths[name] = model_path

        return TrainedModels(models=trained, model_paths=paths)

    def prepare_model_for_deployment(self, model: BaseEstimator) -> tuple[BaseEstimator, dict[str, Any]]:
        X_fit, X_validation, y_fit, y_validation, validation_split_metadata = split_data(
            self.data.X_train,
            self.data.y_train,
            test_size=config.THRESHOLD_VALIDATION_SIZE,
            random_state=config.RANDOM_STATE,
            split_strategy=self.data.split_metadata.get("split_strategy", config.DEFAULT_SPLIT_STRATEGY),
        )

        min_class_count = int(np.min(np.bincount(np.asarray(y_fit, dtype=int))))
        calibration_cv = min(config.CALIBRATION_CV, min_class_count)
        if calibration_cv < 2:
            raise ValueError("Calibration requires at least two samples in each class.")

        calibrated_model = CalibratedClassifierCV(
            estimator=clone(model),
            method=config.CALIBRATION_METHOD,
            cv=calibration_cv,
        )
        calibrated_model.fit(X_fit, y_fit)

        validation_scores = ensure_probabilities(calibrated_model, X_validation)
        validation_metrics = select_decision_threshold(
            y_true=y_validation,
            probabilities=validation_scores,
            metric_name=config.THRESHOLD_SELECTION_METRIC,
        )

        return calibrated_model, {
            "prediction_threshold": validation_metrics["decision_threshold"],
            "threshold_selection_metric": config.THRESHOLD_SELECTION_METRIC,
            "calibration_method": config.CALIBRATION_METHOD,
            "calibration_cv": calibration_cv,
            "threshold_validation_size": config.THRESHOLD_VALIDATION_SIZE,
            "validation_rows": int(len(X_validation)),
            "validation_split_strategy": validation_split_metadata["split_strategy"],
            "validation_metrics": validation_metrics,
        }

    @staticmethod
    def save_best_model(
        model: BaseEstimator,
        model_name: str,
        deployment_info: dict[str, Any] | None = None,
    ) -> Path:
        metadata = deployment_info or {}
        artifact = {
            "artifact_version": 3,
            "model_name": model_name,
            "model": model,
            "prediction_threshold": float(
                metadata.get("prediction_threshold", config.DEFAULT_PREDICTION_THRESHOLD)
            ),
            "threshold_selection_metric": metadata.get("threshold_selection_metric"),
            "calibration_method": metadata.get("calibration_method"),
            "calibration_cv": metadata.get("calibration_cv"),
            "threshold_validation_size": metadata.get("threshold_validation_size"),
            "validation_rows": metadata.get("validation_rows"),
            "validation_split_strategy": metadata.get("validation_split_strategy"),
            "validation_metrics": metadata.get("validation_metrics"),
            "deployment_metrics": metadata.get("deployment_metrics"),
            "split_strategy": metadata.get("split_strategy"),
            "feature_contract": metadata.get("feature_contract"),
            "monitoring_report": metadata.get("monitoring_report"),
            "explainability": metadata.get("explainability"),
        }
        out_path = config.MODELS_DIR / "best_model.pkl"
        joblib.dump(artifact, out_path)
        return out_path


def ensure_probabilities(model: BaseEstimator, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return scores
    return model.predict(X)


def predict_with_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(probabilities) >= threshold).astype(int)


def compute_binary_classification_metrics(y_true, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = predict_with_threshold(probabilities, threshold)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "decision_threshold": float(threshold),
    }


def select_decision_threshold(
    y_true,
    probabilities: np.ndarray,
    metric_name: str = config.THRESHOLD_SELECTION_METRIC,
) -> dict[str, float]:
    supported_metrics = {"accuracy", "precision", "recall", "f1"}
    if metric_name not in supported_metrics:
        raise ValueError(f"Unsupported threshold metric: {metric_name}")

    clipped_probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    candidate_thresholds = np.unique(
        np.concatenate(
            (
                clipped_probabilities,
                np.array([config.DEFAULT_PREDICTION_THRESHOLD], dtype=float),
            )
        )
    )

    best_metrics = compute_binary_classification_metrics(
        y_true,
        clipped_probabilities,
        config.DEFAULT_PREDICTION_THRESHOLD,
    )
    best_score = best_metrics[metric_name]

    for threshold in candidate_thresholds:
        candidate_metrics = compute_binary_classification_metrics(y_true, clipped_probabilities, float(threshold))
        candidate_score = candidate_metrics[metric_name]
        if candidate_score > best_score:
            best_metrics = candidate_metrics
            best_score = candidate_score

    return best_metrics
