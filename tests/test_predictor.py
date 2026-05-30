import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from src.monitoring import build_serving_contract
from src.predictor import load_best_model, predict_from_payload


class FixedProbabilityModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, X):
        rows = len(X)
        positive = np.full(rows, self.probability, dtype=float)
        negative = 1.0 - positive
        return np.column_stack([negative, positive])


class TestPredictor(unittest.TestCase):
    def test_predict_from_payload_uses_persisted_threshold(self):
        with TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "best_model.pkl"
            contract = build_serving_contract(pd.DataFrame([{"feature": 1.0, "segment": "A"}]))
            joblib.dump(
                {
                    "model_name": "fixed_model",
                    "model": FixedProbabilityModel(0.75),
                    "prediction_threshold": 0.8,
                    "threshold_selection_metric": "f1",
                    "calibration_method": "sigmoid",
                    "feature_contract": contract,
                },
                model_path,
            )

            result = predict_from_payload({"feature": 1, "segment": "A"}, model_path=str(model_path))

            self.assertEqual(result["model_name"], "fixed_model")
            self.assertEqual(result["prediction"], 0)
            self.assertEqual(result["probability_canceled"], 0.75)
            self.assertEqual(result["prediction_threshold"], 0.8)
            self.assertEqual(result["threshold_selection_metric"], "f1")
            self.assertEqual(result["calibration_method"], "sigmoid")

    def test_predict_from_payload_rejects_unexpected_or_missing_features(self):
        with TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "best_model.pkl"
            contract = build_serving_contract(pd.DataFrame([{"feature": 1.0, "segment": "A"}]))
            joblib.dump(
                {
                    "model_name": "fixed_model",
                    "model": FixedProbabilityModel(0.75),
                    "feature_contract": contract,
                },
                model_path,
            )

            with self.assertRaisesRegex(ValueError, "Faltan features requeridas"):
                predict_from_payload({"feature": 1}, model_path=str(model_path))

            with self.assertRaisesRegex(ValueError, "Features no esperadas"):
                predict_from_payload({"feature": 1, "segment": "A", "extra": 2}, model_path=str(model_path))

    def test_load_best_model_defaults_threshold_for_legacy_artifact(self):
        with TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "best_model.pkl"
            joblib.dump(
                {
                    "model_name": "legacy_model",
                    "model": FixedProbabilityModel(0.6),
                },
                model_path,
            )

            artifact = load_best_model(str(model_path))

            self.assertEqual(artifact["model_name"], "legacy_model")
            self.assertEqual(artifact["prediction_threshold"], 0.5)
            self.assertIsNone(artifact["threshold_selection_metric"])


if __name__ == "__main__":
    unittest.main()
