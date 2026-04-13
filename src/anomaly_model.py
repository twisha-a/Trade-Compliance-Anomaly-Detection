"""Isolation Forest anomaly detection for trade data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def train_isolation_forest(
    X: pd.DataFrame,
    contamination: float = 0.02,
    n_estimators: int = 100,
    random_state: int = 42,
) -> tuple[IsolationForest, StandardScaler]:
    """Fit a StandardScaler and IsolationForest on the feature matrix.

    Args:
        X: Feature matrix (numeric columns only).
        contamination: Expected fraction of anomalies in the dataset.
        n_estimators: Number of trees in the forest.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (fitted IsolationForest, fitted StandardScaler).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    return model, scaler


def predict_anomalies(
    model: IsolationForest,
    scaler: StandardScaler,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Score and label trades.

    Args:
        model: Fitted IsolationForest.
        scaler: Fitted StandardScaler.
        X: Feature matrix with the same columns used during training.

    Returns:
        Tuple of (labels, scores).
        Labels: ``-1`` = anomaly, ``1`` = normal.
        Scores: raw ``score_samples`` values (lower = more anomalous).
    """
    X_scaled = scaler.transform(X)
    labels = model.predict(X_scaled)
    scores = model.score_samples(X_scaled)
    return labels, scores


def score_trades(
    df: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:
    """Attach anomaly labels and scores to the trade DataFrame.

    Args:
        df: Original trade DataFrame.
        labels: Array returned by :func:`predict_anomalies`.
        scores: Raw anomaly scores (negated to produce a *higher = more anomalous* metric).

    Returns:
        Copy of *df* with new ``is_anomaly`` (bool) and ``anomaly_score`` (float) columns.
    """
    df = df.copy()
    df["is_anomaly"] = labels == -1
    df["anomaly_score"] = -scores  # higher → more anomalous
    return df
