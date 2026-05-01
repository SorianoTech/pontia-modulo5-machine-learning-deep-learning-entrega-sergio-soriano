from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.data_loader import DataBundle

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


@dataclass
class TrainedModels:
    models: Dict[str, Pipeline]
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

        if not self.skip_neural_net:
            estimators["neural_network"] = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                learning_rate_init=0.001,
                max_iter=50 if self.quick_mode else 200,
                early_stopping=True,
                random_state=config.RANDOM_STATE,
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
        trained: Dict[str, Pipeline] = {}
        paths: Dict[str, Path] = {}

        for name, estimator in estimators.items():
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
    def save_best_model(model: Pipeline, model_name: str) -> Path:
        artifact = {
            "model_name": model_name,
            "model": model,
        }
        out_path = config.MODELS_DIR / "best_model.pkl"
        joblib.dump(artifact, out_path)
        return out_path


def ensure_probabilities(model: Pipeline, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return scores
    return model.predict(X)
