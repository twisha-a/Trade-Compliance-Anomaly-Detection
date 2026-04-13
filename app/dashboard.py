"""Streamlit dashboard for Trade Compliance & Anomaly Detection."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow absolute imports from the project root regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import load_trades, validate_trades
from src.feature_engineering import build_feature_matrix
from src.anomaly_model import train_isolation_forest, predict_anomalies, score_trades
from src.compliance_rag import (
    COMPLIANCE_DOCS,
    build_bm25_index,
    build_faiss_index,
    hybrid_retrieve,
)
from src.alert_engine import build_llm, generate_alert

DATA_PATH = Path(__file__).parent.parent / "data" / "synthetic_trades.csv"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trade Compliance Monitor",
    page_icon="📊",
    layout="wide",
)


# ── Cached pipeline ───────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Running anomaly detection pipeline…")
def load_pipeline() -> pd.DataFrame:
    """Load data and run the full anomaly detection pipeline.

    Returns:
        Trade DataFrame enriched with feature columns, ``is_anomaly``,
        and ``anomaly_score``.
    """
    raw = load_trades(DATA_PATH)
    raw = validate_trades(raw)
    features = build_feature_matrix(raw)
    model, scaler = train_isolation_forest(features)
    labels, scores = predict_anomalies(model, scaler, features)
    # Add derived feature columns back for display
    for col in ["notional_value", "trade_velocity", "price_deviation"]:
        if col in features.columns:
            raw[col] = features[col].values
    return score_trades(raw, labels, scores)


@st.cache_resource(show_spinner="Initialising RAG pipeline (embedding rules)…")
def load_rag() -> tuple:
    """Build BM25 + FAISS indices over the compliance corpus.

    Returns:
        (bm25, faiss_index, embeddings, success_flag)
    """
    try:
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        bm25 = build_bm25_index(COMPLIANCE_DOCS)
        faiss_index, _ = build_faiss_index(COMPLIANCE_DOCS, embeddings)
        return bm25, faiss_index, embeddings, True
    except Exception as exc:
        return None, None, None, False


@st.cache_resource(show_spinner="Connecting to Ollama LLM…")
def load_llm() -> tuple:
    """Instantiate the Ollama LLM.

    Returns:
        (llm, success_flag)
    """
    try:
        llm = build_llm()
        return llm, True
    except Exception:
        return None, False


# ── Guard: data file must exist ───────────────────────────────────────────────
if not DATA_PATH.exists():
    st.error(
        f"**Data file not found:** `{DATA_PATH}`\n\n"
        "Generate it first:\n```bash\npython data/synthetic_trades.py\n```"
    )
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_pipeline()
flagged = df[df["is_anomaly"]].copy().reset_index(drop=True)

# ── Header metrics ────────────────────────────────────────────────────────────
st.title("📊 Trade Compliance & Anomaly Detection")
c1, c2, c3 = st.columns(3)
c1.metric("Total Trades", f"{len(df):,}")
c2.metric("Flagged Anomalies", f"{len(flagged):,}")
c3.metric("Anomaly Rate", f"{len(flagged) / len(df):.1%}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🚨 Flagged Trades", "👤 Trader Risk", "📈 Symbol Analysis", "🤖 Compliance Alert"]
)

# ── Tab 1: Flagged Trades ─────────────────────────────────────────────────────
with tab1:
    st.subheader("Flagged Trades")

    display_cols = [
        "symbol", "trader_id", "quantity", "price", "notional_value",
        "order_type", "timestamp", "anomaly_score",
    ]
    available = [c for c in display_cols if c in flagged.columns]

    st.dataframe(
        flagged[available]
        .sort_values("anomaly_score", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
    )

# ── Tab 2: Trader Risk Scores ─────────────────────────────────────────────────
with tab2:
    st.subheader("Trader Risk Scores (anomaly rate per trader)")

    trader_stats = (
        df.groupby("trader_id")
        .agg(total=("is_anomaly", "count"), anomalies=("is_anomaly", "sum"))
        .assign(risk_score=lambda x: x["anomalies"] / x["total"])
        .sort_values("risk_score", ascending=False)
        .head(20)
    )

    st.bar_chart(trader_stats["risk_score"])

# ── Tab 3: Symbol Analysis ────────────────────────────────────────────────────
with tab3:
    st.subheader("Anomaly Count by Symbol")
    sym_counts = flagged.groupby("symbol").size().sort_values(ascending=False)
    st.bar_chart(sym_counts)

# ── Tab 4: Compliance Alert ───────────────────────────────────────────────────
with tab4:
    st.subheader("Per-Trade LLM Compliance Alert")

    if len(flagged) == 0:
        st.info("No anomalies detected.")
    else:
        trade_idx = st.selectbox(
            "Select a flagged trade to analyse",
            options=list(range(len(flagged))),
            format_func=lambda i: (
                f"#{i}  |  {flagged.loc[i, 'symbol']}  |  {flagged.loc[i, 'trader_id']}"
                f"  |  qty={flagged.loc[i, 'quantity']}  |  score={flagged.loc[i, 'anomaly_score']:.4f}"
            ),
        )

        selected = flagged.loc[trade_idx].to_dict()

        with st.expander("Trade Details", expanded=True):
            detail_keys = [
                "symbol", "trader_id", "quantity", "price", "notional_value",
                "order_type", "timestamp", "anomaly_score",
            ]
            st.json({k: str(selected.get(k, "N/A")) for k in detail_keys if k in selected})

        if st.button("Generate Compliance Alert", type="primary"):
            bm25, faiss_index, embeddings, rag_ok = load_rag()
            llm, llm_ok = load_llm()

            if not rag_ok or not llm_ok:
                st.warning(
                    "**Ollama is not running.**  Start it and pull the required models:\n"
                    "```bash\nollama serve\n"
                    "ollama pull qwen3:8b\n"
                    "ollama pull nomic-embed-text\n```"
                )
            else:
                query = (
                    f"suspicious trade {selected.get('symbol', '')} "
                    f"quantity {selected.get('quantity', '')} wash trade volume spike"
                )

                with st.spinner("Retrieving rules and generating alert…"):
                    rules = hybrid_retrieve(
                        query, bm25, faiss_index, embeddings, COMPLIANCE_DOCS
                    )
                    alert = generate_alert(selected, rules, llm)

                st.success("Compliance alert generated.")

                st.write("**Retrieved Rules:**")
                for rule in rules:
                    st.markdown(f"> {rule}")

                st.write("**LLM Compliance Alert:**")
                st.info(alert)
