"""Compute derived features for anomaly detection."""
from __future__ import annotations

import pandas as pd
import numpy as np

FEATURE_COLUMNS: list[str] = [
    "quantity",
    "price",
    "notional_value",
    "trade_velocity",
    "price_deviation",
]


def add_notional_value(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``notional_value = quantity * price``.

    Args:
        df: Trade DataFrame with ``quantity`` and ``price`` columns.

    Returns:
        Copy of *df* with a new ``notional_value`` column.
    """
    df = df.copy()
    df["notional_value"] = df["quantity"] * df["price"]
    return df


def add_trade_velocity(df: pd.DataFrame, window_minutes: int = 60) -> pd.DataFrame:
    """Add ``trade_velocity``: number of trades a trader made in the preceding window.

    Args:
        df: Trade DataFrame with ``trader_id`` and ``timestamp`` columns.
        window_minutes: Rolling time window in minutes.

    Returns:
        Copy of *df* sorted by timestamp with a new ``trade_velocity`` column.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    window = pd.Timedelta(minutes=window_minutes)
    velocities: list[int] = []

    for idx, row in df.iterrows():
        same_trader = df[df["trader_id"] == row["trader_id"]]
        count = int(
            (
                (same_trader["timestamp"] >= row["timestamp"] - window)
                & (same_trader["timestamp"] <= row["timestamp"])
            ).sum()
        )
        velocities.append(count)

    df["trade_velocity"] = velocities
    return df


def add_price_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``price_deviation``: z-score of price relative to per-symbol statistics.

    Args:
        df: Trade DataFrame with ``symbol`` and ``price`` columns.

    Returns:
        Copy of *df* with a new ``price_deviation`` column.
    """
    df = df.copy()
    stats = (
        df.groupby("symbol")["price"]
        .agg(sym_mean="mean", sym_std="std")
        .reset_index()
    )
    df = df.merge(stats, on="symbol", how="left")
    df["sym_std"] = df["sym_std"].fillna(1.0).replace(0.0, 1.0)
    df["price_deviation"] = (df["price"] - df["sym_mean"]) / df["sym_std"]
    df.drop(columns=["sym_mean", "sym_std"], inplace=True)
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering steps and return the feature matrix.

    Args:
        df: Validated trade DataFrame.

    Returns:
        DataFrame with columns defined by :data:`FEATURE_COLUMNS`.
    """
    df = add_notional_value(df)
    df = add_trade_velocity(df)
    df = add_price_deviation(df)
    return df[FEATURE_COLUMNS].reset_index(drop=True)
