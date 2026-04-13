"""Unit tests for the trade compliance pipeline."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic_trades import generate_trades
from src.ingestion import validate_trades
from src.feature_engineering import build_feature_matrix, FEATURE_COLUMNS
from src.anomaly_model import train_isolation_forest, predict_anomalies, score_trades


class TestSyntheticData(unittest.TestCase):
    """Tests for the synthetic trade generator."""

    def setUp(self) -> None:
        self.df = generate_trades(n_trades=200, random_state=0)

    def test_row_count(self) -> None:
        self.assertEqual(len(self.df), 200)

    def test_required_columns_present(self) -> None:
        for col in ("symbol", "trader_id", "quantity", "price", "timestamp", "order_type"):
            self.assertIn(col, self.df.columns, f"Missing column: {col}")

    def test_anomaly_rate_in_range(self) -> None:
        """~2% anomalies expected; must be >0% and <10%."""
        rate = self.df["is_anomaly_label"].mean()
        self.assertGreater(rate, 0.0, "No anomalies injected")
        self.assertLess(rate, 0.10, "Anomaly rate unexpectedly high")

    def test_no_null_values_in_core_fields(self) -> None:
        core = ["symbol", "trader_id", "quantity", "price", "timestamp", "order_type"]
        self.assertEqual(self.df[core].isnull().sum().sum(), 0)

    def test_positive_quantity_and_price(self) -> None:
        self.assertTrue((self.df["quantity"] > 0).all())
        self.assertTrue((self.df["price"] > 0).all())


class TestIngestion(unittest.TestCase):
    """Tests for schema validation in ingestion.py."""

    def setUp(self) -> None:
        self.df = generate_trades(n_trades=100, random_state=1)

    def test_validate_passes_good_data(self) -> None:
        validated = validate_trades(self.df)
        self.assertEqual(len(validated), 100)

    def test_validate_raises_on_missing_column(self) -> None:
        bad = self.df.drop(columns=["symbol"])
        with self.assertRaises(ValueError):
            validate_trades(bad)

    def test_validate_raises_on_null_values(self) -> None:
        bad = self.df.copy()
        bad.loc[0, "price"] = None
        with self.assertRaises(ValueError):
            validate_trades(bad)

    def test_validate_raises_on_negative_quantity(self) -> None:
        bad = self.df.copy()
        bad.loc[0, "quantity"] = -5
        with self.assertRaises(ValueError):
            validate_trades(bad)

    def test_validate_raises_on_zero_price(self) -> None:
        bad = self.df.copy()
        bad.loc[0, "price"] = 0
        with self.assertRaises(ValueError):
            validate_trades(bad)


class TestFeatureEngineering(unittest.TestCase):
    """Tests for feature_engineering.py."""

    def setUp(self) -> None:
        self.df = generate_trades(n_trades=100, random_state=2)

    def test_all_feature_columns_added(self) -> None:
        features = build_feature_matrix(self.df)
        for col in FEATURE_COLUMNS:
            self.assertIn(col, features.columns, f"Missing feature: {col}")

    def test_notional_value_correct(self) -> None:
        features = build_feature_matrix(self.df)
        expected = (self.df["quantity"] * self.df["price"]).values
        np.testing.assert_allclose(
            features["notional_value"].values, expected, rtol=1e-5
        )

    def test_trade_velocity_positive(self) -> None:
        features = build_feature_matrix(self.df)
        self.assertTrue((features["trade_velocity"] >= 1).all())

    def test_no_nulls_in_features(self) -> None:
        features = build_feature_matrix(self.df)
        self.assertEqual(features.isnull().sum().sum(), 0)


class TestAnomalyModel(unittest.TestCase):
    """Tests for anomaly_model.py."""

    def setUp(self) -> None:
        self.raw = generate_trades(n_trades=300, random_state=3)
        self.features = build_feature_matrix(self.raw)

    def test_labels_are_binary(self) -> None:
        model, scaler = train_isolation_forest(self.features, contamination=0.05)
        labels, _ = predict_anomalies(model, scaler, self.features)
        self.assertTrue(set(labels).issubset({-1, 1}))

    def test_anomaly_fraction_near_contamination(self) -> None:
        contamination = 0.05
        model, scaler = train_isolation_forest(
            self.features, contamination=contamination
        )
        labels, _ = predict_anomalies(model, scaler, self.features)
        actual_rate = (labels == -1).mean()
        self.assertAlmostEqual(actual_rate, contamination, delta=0.02)

    def test_score_trades_adds_columns(self) -> None:
        model, scaler = train_isolation_forest(self.features, contamination=0.05)
        labels, scores = predict_anomalies(model, scaler, self.features)
        scored = score_trades(self.raw, labels, scores)
        self.assertIn("is_anomaly", scored.columns)
        self.assertIn("anomaly_score", scored.columns)
        self.assertEqual(scored["is_anomaly"].dtype, bool)

    def test_anomaly_recall_on_injected(self) -> None:
        """Recall on injected anomalies should be >= 0.5 (small dataset)."""
        model, scaler = train_isolation_forest(self.features, contamination=0.05)
        labels, _ = predict_anomalies(model, scaler, self.features)
        true_anomaly = self.raw["is_anomaly_label"].values
        if true_anomaly.sum() == 0:
            self.skipTest("No injected anomalies in this sample")
        pred_anomaly = (labels == -1).astype(int)
        recall = float((pred_anomaly & true_anomaly).sum()) / float(true_anomaly.sum())
        self.assertGreaterEqual(recall, 0.5, f"Recall too low: {recall:.2f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
