from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False

from src import config
from src.data_loader import DataBundle

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

    @staticmethod
    def save_best_model(model: BaseEstimator, model_name: str) -> Path:
        artifact = {
            "model_name": model_name,
            "model": model,
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
