from __future__ import annotations

from typing import Any

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src import config


def _random_forest_space() -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    grid = {
        "classifier__n_estimators": [150, 250, 400],
        "classifier__max_depth": [None, 8, 14],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"],
    }

    randomized = {
        "classifier__n_estimators": [120, 180, 240, 320, 420],
        "classifier__max_depth": [None, 6, 10, 14, 20],
        "classifier__min_samples_split": [2, 4, 6, 8, 10],
        "classifier__min_samples_leaf": [1, 2, 3, 4],
        "classifier__max_features": ["sqrt", "log2", None],
    }
    return grid, randomized


def tune_pipeline(
    pipeline: Pipeline,
    X_train,
    y_train,
    method: str = config.DEFAULT_TUNING_METHOD,
    cv: int = config.DEFAULT_TUNING_CV,
    n_iter: int = config.DEFAULT_TUNING_ITER,
    scoring: str = config.PRIMARY_METRIC,
) -> tuple[Pipeline, dict[str, Any]]:
    method = method.lower().strip()
    grid_space, random_space = _random_forest_space()

    if method == "grid":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid_space,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )
    elif method == "randomized":
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=random_space,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
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
    }

    return search.best_estimator_, metadata
