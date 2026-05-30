import unittest

import pandas as pd

from src.data_loader import resolve_split_strategy, split_data


class TestDataLoader(unittest.TestCase):
    def test_auto_split_prefers_chronological_when_booking_period_exists(self):
        X = pd.DataFrame(
            {
                "arrival_date_year": [2015, 2015, 2016, 2016],
                "arrival_date_month": ["January", "February", "January", "February"],
                "feature": [1, 2, 3, 4],
            }
        )

        self.assertEqual(resolve_split_strategy(X, "auto"), "chronological")

    def test_chronological_split_keeps_future_rows_in_test(self):
        X = pd.DataFrame(
            {
                "arrival_date_year": [2015, 2015, 2016, 2016, 2017],
                "arrival_date_month": ["January", "February", "January", "February", "January"],
                "feature": [1, 2, 3, 4, 5],
            }
        )
        y = pd.Series([0, 1, 0, 1, 1])

        X_train, X_test, _, _, metadata = split_data(X, y, test_size=0.4, split_strategy="chronological")

        self.assertEqual(metadata["split_strategy"], "chronological")
        self.assertLessEqual(metadata["train_period_end"], metadata["test_period_start"])
        self.assertEqual(metadata["train_period_start"], "2015-01-01")
        self.assertEqual(metadata["test_period_end"], "2017-01-01")


if __name__ == "__main__":
    unittest.main()
