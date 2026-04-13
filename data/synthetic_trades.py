"""Generate synthetic trade data with ~2% injected anomalies (wash trades + volume spikes)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_trades(n_trades: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Return a DataFrame of synthetic trade records with labelled anomalies.

    Args:
        n_trades: Total number of trades to generate.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with columns: symbol, trader_id, quantity, price, timestamp,
        order_type, is_anomaly_label, anomaly_type.
    """
    rng = np.random.default_rng(random_state)

    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM"]
    traders = [f"T{i:03d}" for i in range(1, 21)]
    order_types = ["BUY", "SELL"]

    base_prices: dict[str, float] = {
        "AAPL": 175.0,
        "GOOGL": 140.0,
        "MSFT": 380.0,
        "AMZN": 185.0,
        "TSLA": 250.0,
        "META": 500.0,
        "NVDA": 800.0,
        "JPM": 195.0,
    }

    n_normal = int(n_trades * 0.98)
    n_anomaly = n_trades - n_normal

    # ── Normal trades ────────────────────────────────────────────────────────
    sym_normal = rng.choice(symbols, size=n_normal)
    trd_normal = rng.choice(traders, size=n_normal)
    qty_normal = rng.integers(10, 500, size=n_normal)
    prc_normal = np.array(
        [base_prices[s] * rng.uniform(0.97, 1.03) for s in sym_normal]
    )
    ts_normal = pd.date_range("2024-01-01", periods=n_normal, freq="1min")
    ord_normal = rng.choice(order_types, size=n_normal)

    # ── Anomalies ────────────────────────────────────────────────────────────
    n_volume = n_anomaly // 2
    n_wash = n_anomaly - n_volume

    sym_anom = rng.choice(symbols, size=n_anomaly)
    trd_anom = rng.choice(traders, size=n_anomaly)

    # Volume spikes: 10× normal quantity
    qty_volume = rng.integers(5_000, 10_000, size=n_volume)
    prc_volume = np.array(
        [base_prices[s] * rng.uniform(0.97, 1.03) for s in sym_anom[:n_volume]]
    )
    ts_volume = pd.date_range("2024-01-15", periods=n_volume, freq="5min")
    ord_volume = rng.choice(order_types, size=n_volume)

    # Wash trades: tiny price movement, rapid cycling
    qty_wash = rng.integers(100, 200, size=n_wash)
    prc_wash = np.array(
        [base_prices[s] * rng.uniform(0.999, 1.001) for s in sym_anom[n_volume:]]
    )
    ts_wash = pd.date_range("2024-01-20", periods=n_wash, freq="2min")
    ord_wash = rng.choice(order_types, size=n_wash)

    # ── Assemble ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "symbol": list(sym_normal) + list(sym_anom),
            "trader_id": list(trd_normal) + list(trd_anom),
            "quantity": list(qty_normal)
            + list(qty_volume)
            + list(qty_wash),
            "price": list(prc_normal) + list(prc_volume) + list(prc_wash),
            "timestamp": list(ts_normal) + list(ts_volume) + list(ts_wash),
            "order_type": list(ord_normal) + list(ord_volume) + list(ord_wash),
            "is_anomaly_label": [0] * n_normal + [1] * n_anomaly,
            "anomaly_type": ["normal"] * n_normal
            + ["volume_spike"] * n_volume
            + ["wash_trade"] * n_wash,
        }
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


if __name__ == "__main__":
    output_path = Path(__file__).parent / "synthetic_trades.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_trades(n_trades=1000)
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} trades → {output_path}")
    print(
        f"Anomalies: {df['is_anomaly_label'].sum()} ({df['is_anomaly_label'].mean():.1%})"
    )
    print(df["anomaly_type"].value_counts())
