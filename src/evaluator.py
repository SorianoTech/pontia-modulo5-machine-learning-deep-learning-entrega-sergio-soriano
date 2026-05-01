from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src import config
from src.model_trainer import ensure_probabilities


class Evaluator:
    def __init__(self, y_true: pd.Series, X_test: pd.DataFrame, models: Dict[str, Pipeline]) -> None:
        self.y_true = y_true
        self.X_test = X_test
        self.models = models

    def evaluate(self) -> pd.DataFrame:
        rows = []
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_prob = ensure_probabilities(model, self.X_test)

            rows.append(
                {
                    "model": name,
                    "accuracy": accuracy_score(self.y_true, y_pred),
                    "precision": precision_score(self.y_true, y_pred, zero_division=0),
                    "recall": recall_score(self.y_true, y_pred, zero_division=0),
                    "f1": f1_score(self.y_true, y_pred, zero_division=0),
                    "roc_auc": roc_auc_score(self.y_true, y_prob),
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
            y_pred = model.predict(self.X_test)
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
