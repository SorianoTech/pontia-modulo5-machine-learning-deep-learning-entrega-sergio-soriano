import unittest

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.tuning import tune_pipeline


class TestTuning(unittest.TestCase):
    def test_tune_pipeline_supports_logistic_regression(self):
        X = pd.DataFrame(
            {
                "f1": [0.1, 0.2, 0.3, 0.8, 0.9, 1.0],
                "f2": [1.0, 0.9, 0.8, 0.2, 0.1, 0.0],
            }
        )
        y = pd.Series([0, 0, 0, 1, 1, 1])
        pipeline = Pipeline(
            [
                ("classifier", LogisticRegression(max_iter=500, random_state=42)),
            ]
        )

        tuned_model, metadata = tune_pipeline(
            pipeline=pipeline,
            model_name="logistic_regression",
            X_train=X,
            y_train=y,
            method="randomized",
            cv=2,
            n_iter=2,
        )

        self.assertIsNotNone(tuned_model)
        self.assertEqual(metadata["model_family"], "logistic_regression")
        self.assertIn("classifier__C", metadata["best_params"])


if __name__ == "__main__":
    unittest.main()
