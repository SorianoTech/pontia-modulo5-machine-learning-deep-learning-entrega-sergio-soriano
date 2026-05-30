import unittest

import numpy as np

from src.model_trainer import select_decision_threshold


class TestModelTrainer(unittest.TestCase):
    def test_select_decision_threshold_optimizes_f1_over_default_threshold(self):
        y_true = np.array([0, 0, 1, 1])
        probabilities = np.array([0.2, 0.7, 0.8, 0.9])

        result = select_decision_threshold(y_true, probabilities, metric_name="f1")

        self.assertEqual(result["decision_threshold"], 0.8)
        self.assertEqual(result["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
