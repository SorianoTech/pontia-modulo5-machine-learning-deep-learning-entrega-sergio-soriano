from __future__ import annotations

import argparse

from src import config
from src.data_loader import build_data_bundle
from src.evaluator import Evaluator
from src.model_trainer import ModelTrainer


def run_pipeline(data_path: str | None = None, quick_mode: bool = False, skip_neural_net: bool = False):
    bundle = build_data_bundle(data_path)

    trainer = ModelTrainer(bundle, quick_mode=quick_mode, skip_neural_net=skip_neural_net)
    trained = trainer.train_all()

    evaluator = Evaluator(bundle.y_test, bundle.X_test, trained.models)
    metrics_df = evaluator.evaluate()
    evaluator.plot_roc_curves()
    evaluator.plot_confusion_matrices()
    evaluator.plot_feature_importance("random_forest")

    best_row = metrics_df.iloc[0]
    best_name = str(best_row["model"])
    best_model = trained.models[best_name]
    best_model_path = trainer.save_best_model(best_model, best_name)

    return {
        "best_model": best_name,
        "best_model_path": str(best_model_path),
        "metrics_path": str(config.METRICS_PATH),
        "metrics": metrics_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate binary classification models")
    parser.add_argument("--data-path", type=str, default=None, help="Custom CSV data path")
    parser.add_argument("--quick", action="store_true", help="Quick training mode")
    parser.add_argument("--skip-neural-net", action="store_true", help="Skip MLP model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(args.data_path, args.quick, args.skip_neural_net)

    print("Training pipeline completed")
    print(f"Best model: {result['best_model']}")
    print(f"Best model artifact: {result['best_model_path']}")
    print(f"Metrics saved at: {result['metrics_path']}")
    print(result["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
