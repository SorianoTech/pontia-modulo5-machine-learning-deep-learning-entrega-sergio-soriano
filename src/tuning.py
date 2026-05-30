from __future__ import annotations

from typing import Any
from itertools import product

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src import config


def _prefix_search_space(space: dict[str, list[Any]], prefix: str) -> dict[str, list[Any]]:
    return {f"{prefix}{key}": values for key, values in space.items()}


def _space_cardinality(space: dict[str, list[Any]]) -> int:
    if not space:
        return 0
    return len(list(product(*space.values())))


def _random_forest_space() -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    grid = {
        "n_estimators": [150, 250, 400],
        "max_depth": [None, 8, 14],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    randomized = {
        "n_estimators": [120, 180, 240, 320, 420],
        "max_depth": [None, 6, 10, 14, 20],
        "min_samples_split": [2, 4, 6, 8, 10],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": ["sqrt", "log2", None],
    }
    return grid, randomized


def _logistic_regression_space() -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    grid = {
        "C": [0.1, 0.5, 1.0, 2.0],
        "class_weight": [None, "balanced"],
    }
    randomized = {
        "C": [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
        "class_weight": [None, "balanced"],
    }
    return grid, randomized


def _decision_tree_space() -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    grid = {
        "max_depth": [4, 8, 12, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": [None, "balanced"],
    }
    randomized = {
        "max_depth": [4, 6, 8, 10, 12, None],
        "min_samples_split": [2, 4, 6, 8, 10],
        "min_samples_leaf": [1, 2, 3, 4],
        "class_weight": [None, "balanced"],
    }
    return grid, randomized


def _xgboost_space() -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    grid = {
        "n_estimators": [80, 140, 220],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.8, 0.9],
    }
    randomized = {
        "n_estimators": [60, 100, 140, 180, 240],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
    }
    return grid, randomized


def _gradient_boosting_space() -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    grid = {
        "n_estimators": [80, 140, 220],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "subsample": [0.8, 0.9, 1.0],
    }
    randomized = {
        "n_estimators": [60, 100, 140, 180, 240],
        "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
        "max_depth": [2, 3, 4, 5],
        "subsample": [0.7, 0.8, 0.9, 1.0],
    }
    return grid, randomized


def _catboost_space() -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    grid = {
        "iterations": [80, 120, 180],
        "learning_rate": [0.03, 0.05, 0.1],
        "depth": [4, 6, 8],
    }
    randomized = {
        "iterations": [60, 100, 140, 180, 240],
        "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
        "depth": [4, 5, 6, 7, 8],
    }
    return grid, randomized


def _neural_network_space() -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    grid = {
        "epochs": [10, 20, 30],
        "batch_size": [128, 256],
        "learning_rate": [0.0005, 0.001, 0.002],
    }
    randomized = {
        "epochs": [8, 10, 15, 20, 30],
        "batch_size": [64, 128, 256],
        "learning_rate": [0.0003, 0.0005, 0.001, 0.002, 0.003],
    }
    return grid, randomized


def _search_spaces_for_model(pipeline, model_name: str) -> tuple[dict[str, list[Any]], dict[str, list[Any]], str]:
    estimator = pipeline.named_steps["classifier"] if hasattr(pipeline, "named_steps") else pipeline
    prefix = "classifier__" if hasattr(pipeline, "named_steps") else ""

    if model_name == "random_forest":
        grid, randomized = _random_forest_space()
        return _prefix_search_space(grid, prefix), _prefix_search_space(randomized, prefix), "random_forest"
    if model_name == "logistic_regression":
        grid, randomized = _logistic_regression_space()
        return _prefix_search_space(grid, prefix), _prefix_search_space(randomized, prefix), "logistic_regression"
    if model_name == "decision_tree":
        grid, randomized = _decision_tree_space()
        return _prefix_search_space(grid, prefix), _prefix_search_space(randomized, prefix), "decision_tree"
    if model_name == "gradient_boosting":
        if estimator.__class__.__name__ == "XGBClassifier":
            grid, randomized = _xgboost_space()
            return _prefix_search_space(grid, prefix), _prefix_search_space(randomized, prefix), "xgboost"
        grid, randomized = _gradient_boosting_space()
        return _prefix_search_space(grid, prefix), _prefix_search_space(randomized, prefix), "gradient_boosting"
    if model_name == "catboost":
        grid, randomized = _catboost_space()
        return _prefix_search_space(grid, prefix), _prefix_search_space(randomized, prefix), "catboost"
    if model_name == "neural_network":
        grid, randomized = _neural_network_space()
        return _prefix_search_space(grid, prefix), _prefix_search_space(randomized, prefix), "neural_network"

    raise ValueError(f"No tuning space configured for model '{model_name}'")


def tune_pipeline(
    pipeline,
    model_name: str,
    X_train,
    y_train,
    method: str = config.DEFAULT_TUNING_METHOD,
    cv: int = config.DEFAULT_TUNING_CV,
    n_iter: int = config.DEFAULT_TUNING_ITER,
    scoring: str = config.PRIMARY_METRIC,
) -> tuple[Pipeline, dict[str, Any]]:
    method = method.lower().strip()
    grid_space, random_space, model_family = _search_spaces_for_model(pipeline, model_name)
    search_jobs = 1 if model_family == "neural_network" else -1

    if method == "grid":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid_space,
            scoring=scoring,
            cv=cv,
            n_jobs=search_jobs,
            verbose=0,
            error_score="raise",
        )
    elif method == "randomized":
        total_candidates = _space_cardinality(random_space)
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=random_space,
            n_iter=min(n_iter, total_candidates) if total_candidates else n_iter,
            scoring=scoring,
            cv=cv,
            random_state=config.RANDOM_STATE,
            n_jobs=search_jobs,
            verbose=0,
            error_score="raise",
        )
    else:
        raise ValueError("tuning_method debe ser 'grid' o 'randomized'")

    search.fit(X_train, y_train)

    metadata = {
        "method": method,
        "cv": cv,
        "n_iter": n_iter if method == "randomized" else None,
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "model_family": model_family,
    }

    return search.best_estimator_, metadata
