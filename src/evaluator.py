from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src import config
from src.model_trainer import (
    compute_binary_classification_metrics,
    ensure_probabilities,
    predict_with_threshold,
)


class Evaluator:
    def __init__(
        self,
        y_true: pd.Series,
        X_test: pd.DataFrame,
        models: Dict[str, object],
        threshold_overrides: Dict[str, float] | None = None,
    ) -> None:
        self.y_true = y_true
        self.X_test = X_test
        self.models = models
        self.threshold_overrides = threshold_overrides or {}

    def evaluate(self) -> pd.DataFrame:
        rows = []
        for name, model in self.models.items():
            y_prob = ensure_probabilities(model, self.X_test)
            threshold = self.threshold_overrides.get(name, config.DEFAULT_PREDICTION_THRESHOLD)
            metrics = compute_binary_classification_metrics(self.y_true, y_prob, threshold)

            rows.append(
                {
                    "model": name,
                    **metrics,
                }
            )

        df = pd.DataFrame(rows).sort_values(by=config.PRIMARY_METRIC, ascending=False)
        df.to_csv(config.METRICS_PATH, index=False)
        return df

    def plot_roc_curves(self, output_path: Path | None = None) -> Path:
        path = output_path if output_path else config.OUTPUTS_DIR / "roc_curves.png"

        plt.figure(figsize=(10, 7))
        for name, model in self.models.items():
            y_prob = ensure_probabilities(model, self.X_test)
            fpr, tpr, _ = roc_curve(self.y_true, y_prob)
            auc = roc_auc_score(self.y_true, y_prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves - Model Comparison")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path

    def plot_confusion_matrices(self) -> list[Path]:
        outputs = []
        for name, model in self.models.items():
            y_prob = ensure_probabilities(model, self.X_test)
            threshold = self.threshold_overrides.get(name, config.DEFAULT_PREDICTION_THRESHOLD)
            y_pred = predict_with_threshold(y_prob, threshold)
            cm = confusion_matrix(self.y_true, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(f"Confusion Matrix - {name}")
            fig.tight_layout()
            path = config.OUTPUTS_DIR / f"confusion_{name}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            outputs.append(path)
        return outputs

    def plot_feature_importance(self, model_name: str = "random_forest") -> Path | None:
        if model_name not in self.models:
            return None

        model = self.models[model_name]
        if not hasattr(model, "named_steps"):
            return None

        classifier = model.named_steps["classifier"]
        if not hasattr(classifier, "feature_importances_"):
            return None

        preprocessor = model.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_

        # Plot top N to keep the chart readable.
        top_n = 20
        idx = np.argsort(importances)[-top_n:]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(np.array(feature_names)[idx], importances[idx])
        ax.set_title(f"Top {top_n} Feature Importances - {model_name}")
        ax.set_xlabel("Importance")
        fig.tight_layout()

        path = config.OUTPUTS_DIR / f"feature_importance_{model_name}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return path

    def plot_permutation_importance(self, model_name: str) -> tuple[Path, Path] | None:
        if model_name not in self.models:
            return None

        result = permutation_importance(
            self.models[model_name],
            self.X_test,
            self.y_true,
            n_repeats=config.PERMUTATION_IMPORTANCE_REPEATS,
            random_state=config.RANDOM_STATE,
            scoring=config.PRIMARY_METRIC,
            n_jobs=1,
        )

        importance_df = (
            pd.DataFrame(
                {
                    "feature": self.X_test.columns,
                    "importance_mean": result.importances_mean,
                    "importance_std": result.importances_std,
                }
            )
            .sort_values(by="importance_mean", ascending=False)
            .reset_index(drop=True)
        )

        csv_path = config.OUTPUTS_DIR / f"permutation_importance_{model_name}.csv"
        importance_df.to_csv(csv_path, index=False)

        top_df = (
            importance_df.head(config.EXPLAINABILITY_TOP_N)
            .sort_values(by="importance_mean", ascending=True)
            .reset_index(drop=True)
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(top_df["feature"], top_df["importance_mean"], xerr=top_df["importance_std"])
        ax.set_title(f"Permutation Importance - {model_name}")
        ax.set_xlabel("Importance mean")
        fig.tight_layout()

        png_path = config.OUTPUTS_DIR / f"permutation_importance_{model_name}.png"
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
        return png_path, csv_path
