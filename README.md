# Trade Compliance & Anomaly Detection System

An end-to-end ML system that detects suspicious trades (wash trades, volume spikes) using Isolation Forest, retrieves relevant SEC/FINRA rules via hybrid BM25 + FAISS RAG, and generates natural-language compliance alerts with a local LLM — all surfaced through an interactive Streamlit dashboard.

---

## Features

- **Synthetic trade generation** — 1 000 trades with ~2% injected anomalies (wash trades + volume spikes)
- **Feature engineering** — notional value, trade velocity (rolling 60-min window), per-symbol price z-score
- **Anomaly detection** — Isolation Forest with StandardScaler; contamination tuned to dataset anomaly rate
- **Hybrid RAG** — BM25 keyword retrieval + FAISS cosine similarity over SEC Rule 10b-5 and FINRA wash-trade rules, embedded with `nomic-embed-text`
- **LLM compliance alerts** — flagged trades queried against retrieved rules; `qwen3:8b` via Ollama generates a 3–5 sentence compliance memo
- **Streamlit dashboard** — flagged trade table, trader risk scores, symbol-level bar charts, per-trade LLM alert on click

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| ML | Scikit-learn (Isolation Forest, StandardScaler) |
| Embeddings | nomic-embed-text via Ollama |
| Vector search | FAISS (faiss-cpu) |
| Keyword search | BM25 (rank-bm25) |
| LLM | Qwen3:8b via Ollama + LangChain |
| Dashboard | Streamlit |
| Data | Pandas, NumPy |

---

## Project Structure

```
trade-compliance-ml/
├── data/
│   ├── __init__.py
│   └── synthetic_trades.py      # generates synthetic_trades.csv
├── src/
│   ├── __init__.py
│   ├── ingestion.py             # load + validate CSV
│   ├── feature_engineering.py  # notional value, velocity, price deviation
│   ├── anomaly_model.py         # Isolation Forest pipeline
│   ├── compliance_rag.py        # BM25 + FAISS hybrid retrieval
│   └── alert_engine.py          # LLM alert generation
├── app/
│   └── dashboard.py             # Streamlit dashboard
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py         # unit tests
├── requirements.txt
└── README.md
```

---

## Results

| Metric | Value |
|---|---|
| Dataset size | 1 000 trades |
| Injected anomaly rate | ~2% (10 volume spikes + 10 wash trades) |
| IsolationForest contamination | 0.02 |
| Anomaly recall (on injected labels) | >= 80% |
| RAG corpus | 6 SEC/FINRA rule documents |
| Retrieval | Hybrid BM25 + cosine (50/50 weight), top-3 |

---

## Run Instructions

### 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

```bash
ollama serve                    # start Ollama server
ollama pull qwen3:8b            # LLM for compliance alerts
ollama pull nomic-embed-text    # embedding model for RAG
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate synthetic data

```bash
python data/synthetic_trades.py
```

This writes `data/synthetic_trades.csv` (1 000 rows, ~2% anomalies).

### 4. Run tests

```bash
python -m pytest tests/ -v
```

### 5. Launch the dashboard

```bash
streamlit run app/dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **Note:** Tabs 1-3 (trade table, risk scores, symbol chart) work without Ollama.
> The **Compliance Alert** tab requires `ollama serve` + both models pulled.

---

## How It Works

1. `data/synthetic_trades.py` generates trades with controlled anomalies seeded for reproducibility.
2. `src/ingestion.py` validates the CSV schema and data types at load time.
3. `src/feature_engineering.py` computes `notional_value`, `trade_velocity`, and `price_deviation` — features that distinguish anomalous patterns.
4. `src/anomaly_model.py` fits a StandardScaler + IsolationForest and returns per-trade anomaly scores.
5. `src/compliance_rag.py` maintains BM25 and FAISS indices over 6 hard-coded SEC/FINRA rule documents. Retrieval combines both scores 50/50.
6. `src/alert_engine.py` formats a prompt with trade details + retrieved rules and calls Qwen3:8b for the final alert.
7. `app/dashboard.py` orchestrates everything with Streamlit caching to avoid re-running the pipeline on every interaction.
