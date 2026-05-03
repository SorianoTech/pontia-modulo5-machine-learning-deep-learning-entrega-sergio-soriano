from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

from src import config
from src.data_loader import build_data_bundle
from src.evaluator import Evaluator
from src.mlflow_tracker import MLflowTracker
from src.model_trainer import ModelTrainer
from src.tuning import tune_pipeline


def _to_float_metrics(row: dict) -> dict[str, float]:
    out = {}
    for key, value in row.items():
        if key == "model":
            continue
        out[key] = float(value)
    return out


def run_pipeline(
    data_path: str | None = None,
    quick_mode: bool = False,
    skip_neural_net: bool = False,
    enable_tuning: bool = False,
    tuning_method: str = config.DEFAULT_TUNING_METHOD,
    tuning_cv: int = config.DEFAULT_TUNING_CV,
    tuning_iter: int = config.DEFAULT_TUNING_ITER,
    enable_mlflow: bool = True,
):
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

    tuning_result = None
    tuned_model_name = f"{best_name}_tuned"
    if enable_tuning and best_name == "random_forest":
        tuned_model, tuning_result = tune_pipeline(
            pipeline=best_model,
            X_train=bundle.X_train,
            y_train=bundle.y_train,
            method=tuning_method,
            cv=tuning_cv,
            n_iter=tuning_iter,
            scoring=config.PRIMARY_METRIC,
        )

        trained.models[tuned_model_name] = tuned_model
        tuned_eval = Evaluator(bundle.y_test, bundle.X_test, {tuned_model_name: tuned_model}).evaluate()
        metrics_df = (
            metrics_df[metrics_df["model"] != tuned_model_name]
            .pipe(lambda df: df if df.empty else df)
            .reset_index(drop=True)
        )
        metrics_df = (
            pd.concat([metrics_df, tuned_eval], ignore_index=True)
            .sort_values(by=config.PRIMARY_METRIC, ascending=False)
            .reset_index(drop=True)
        )
        metrics_df.to_csv(config.METRICS_PATH, index=False)

        best_row = metrics_df.iloc[0]
        best_name = str(best_row["model"])
        best_model = trained.models[best_name]

    best_model_path = trainer.save_best_model(best_model, best_name)

    mlflow_run_id = None
    if enable_mlflow:
        tracker = MLflowTracker()
        run_name = f"pipeline-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        with tracker.start_run(
            run_name=run_name,
            tags={"project": "hotel-cancellation", "pipeline": "baseline+bonus-phase2"},
        ) as run:
            mlflow_run_id = run.info.run_id
            tracker.log_params(
                {
                    "quick_mode": quick_mode,
                    "skip_neural_net": skip_neural_net,
                    "enable_tuning": enable_tuning,
                    "tuning_method": tuning_method,
                    "tuning_cv": tuning_cv,
                    "tuning_iter": tuning_iter,
                    "best_model": best_name,
                }
            )

            for row in metrics_df.to_dict(orient="records"):
                model_name = row["model"]
                numeric_metrics = _to_float_metrics(row)
                tracker.log_metrics({f"{model_name}_{k}": v for k, v in numeric_metrics.items()})

            if tuning_result:
                best_tuning_score = tuning_result.get("best_score")
                if best_tuning_score is not None:
                    tracker.log_metrics({"tuning_best_cv_score": float(best_tuning_score)})
                tracker.log_params({
                    "tuning_best_params": str(tuning_result.get("best_params", {})),
                })

            tracker.log_artifact(config.METRICS_PATH)
            roc_path = config.OUTPUTS_DIR / "roc_curves.png"
            if roc_path.exists():
                tracker.log_artifact(roc_path)

            fi_path = config.OUTPUTS_DIR / "feature_importance_random_forest.png"
            if fi_path.exists():
                tracker.log_artifact(fi_path)

            tracker.log_model(best_model, input_example=bundle.X_train.head(5), artifact_path="best_model")

    return {
        "best_model": best_name,
        "best_model_path": str(best_model_path),
        "metrics_path": str(config.METRICS_PATH),
        "metrics": metrics_df,
        "tuning": tuning_result,
        "mlflow_run_id": mlflow_run_id,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate binary classification models")
    parser.add_argument("--data-path", type=str, default=None, help="Custom CSV data path")
    parser.add_argument("--quick", action="store_true", help="Quick training mode")
    parser.add_argument("--skip-neural-net", action="store_true", help="Skip MLP model")
    parser.add_argument("--enable-tuning", action="store_true", help="Enable hyperparameter tuning")
    parser.add_argument(
        "--tuning-method",
        type=str,
        default=config.DEFAULT_TUNING_METHOD,
        choices=["grid", "randomized"],
        help="Hyperparameter search method",
    )
    parser.add_argument("--tuning-cv", type=int, default=config.DEFAULT_TUNING_CV, help="Cross-validation folds")
    parser.add_argument(
        "--tuning-iter",
        type=int,
        default=config.DEFAULT_TUNING_ITER,
        help="Iterations for RandomizedSearchCV",
    )
    parser.add_argument("--disable-mlflow", action="store_true", help="Disable MLflow tracking for this run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(
        data_path=args.data_path,
        quick_mode=args.quick,
        skip_neural_net=args.skip_neural_net,
        enable_tuning=args.enable_tuning,
        tuning_method=args.tuning_method,
        tuning_cv=args.tuning_cv,
        tuning_iter=args.tuning_iter,
        enable_mlflow=not args.disable_mlflow,
    )

    print("Training pipeline completed")
    print(f"Best model: {result['best_model']}")
    print(f"Best model artifact: {result['best_model_path']}")
    print(f"Metrics saved at: {result['metrics_path']}")
    print(f"MLflow run id: {result['mlflow_run_id']}")
    if result["tuning"]:
        print(f"Tuning summary: {result['tuning']}")
    print(result["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
