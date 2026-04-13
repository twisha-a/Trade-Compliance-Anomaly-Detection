"""Generate LLM compliance alert summaries via Ollama Qwen3:8b."""
from __future__ import annotations

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

_ALERT_PROMPT = PromptTemplate.from_template(
    """You are a financial compliance officer reviewing flagged trades.

Trade flagged as anomalous:
  Symbol      : {symbol}
  Trader ID   : {trader_id}
  Quantity    : {quantity}
  Price       : ${price:.2f}
  Notional    : ${notional_value:.2f}
  Order Type  : {order_type}
  Timestamp   : {timestamp}
  Anomaly Score: {anomaly_score:.4f}  (higher = more suspicious)

Relevant compliance rules:
{rules}

Write a concise compliance alert (3–5 sentences) that:
1. Explains why this trade is suspicious.
2. Identifies which specific rule(s) may be violated.
3. Recommends next steps for the compliance team.

Alert:"""
)


def build_llm(
    model: str = "qwen3:8b",
    base_url: str = "http://localhost:11434",
) -> OllamaLLM:
    """Instantiate the Ollama LLM client.

    Args:
        model: Ollama model name (default ``qwen3:8b``).
        base_url: Ollama server URL.

    Returns:
        Configured :class:`OllamaLLM` instance.
    """
    return OllamaLLM(model=model, base_url=base_url)


def generate_alert(
    trade: dict,
    rules: list[str],
    llm: OllamaLLM,
) -> str:
    """Generate a natural-language compliance alert for a flagged trade.

    Args:
        trade: Dict of trade field values (must include keys used in the prompt template).
        rules: Retrieved compliance rule strings from the RAG pipeline.
        llm: Fitted :class:`OllamaLLM` instance.

    Returns:
        Compliance alert string, or a human-readable error if Ollama is unavailable.
    """
    rules_text = "\n\n".join(f"• {r}" for r in rules)

    # Ensure required numeric fields have defaults to avoid format errors
    safe_trade = {
        "symbol": trade.get("symbol", "N/A"),
        "trader_id": trade.get("trader_id", "N/A"),
        "quantity": float(trade.get("quantity", 0)),
        "price": float(trade.get("price", 0)),
        "notional_value": float(trade.get("notional_value", 0)),
        "order_type": trade.get("order_type", "N/A"),
        "timestamp": str(trade.get("timestamp", "N/A")),
        "anomaly_score": float(trade.get("anomaly_score", 0)),
    }

    prompt = _ALERT_PROMPT.format(rules=rules_text, **safe_trade)

    try:
        return llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        return (
            "[Ollama unavailable — start the server and pull the required models]\n"
            "  ollama serve\n"
            "  ollama pull qwen3:8b\n"
            "  ollama pull nomic-embed-text\n\n"
            f"Error: {exc}"
        )
