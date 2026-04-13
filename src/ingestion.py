"""Load and validate trade data from a CSV file."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"symbol", "trader_id", "quantity", "price", "timestamp", "order_type"}
)
NUMERIC_COLUMNS: list[str] = ["quantity", "price"]


def load_trades(path: str | Path) -> pd.DataFrame:
    """Read a trade CSV from disk.

    Args:
        path: Path to the CSV file.

    Returns:
        Raw DataFrame with ``timestamp`` parsed as datetime.
    """
    return pd.read_csv(path, parse_dates=["timestamp"])


def validate_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, types, and value constraints.

    Args:
        df: Raw trade DataFrame.

    Returns:
        The same DataFrame if valid.

    Raises:
        ValueError: If any validation check fails.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    null_counts = df[list(REQUIRED_COLUMNS)].isnull().sum()
    if null_counts.any():
        bad = null_counts[null_counts > 0]
        raise ValueError(f"Null values found:\n{bad}")

    for col in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric, got {df[col].dtype}")

    if (df["quantity"] <= 0).any():
        raise ValueError("All 'quantity' values must be positive.")
    if (df["price"] <= 0).any():
        raise ValueError("All 'price' values must be positive.")

    return df
