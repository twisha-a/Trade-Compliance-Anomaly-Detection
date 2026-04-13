"""Hybrid BM25 + FAISS retrieval pipeline over SEC/FINRA compliance rules."""
from __future__ import annotations

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from langchain_ollama import OllamaEmbeddings

# ── Compliance rule corpus ────────────────────────────────────────────────────

COMPLIANCE_DOCS: list[str] = [
    # SEC Rule 10b-5
    (
        "SEC Rule 10b-5 (a): It is unlawful for any person, directly or indirectly, "
        "to employ any device, scheme, or artifice to defraud in connection with the "
        "purchase or sale of any security registered on a national securities exchange."
    ),
    (
        "SEC Rule 10b-5 (b): It is unlawful to make any untrue statement of a material "
        "fact or to omit to state a material fact necessary in order to make the statements "
        "made, in light of the circumstances under which they were made, not misleading."
    ),
    (
        "SEC Rule 10b-5 (c): It is unlawful to engage in any act, practice, or course of "
        "business which operates or would operate as a fraud or deceit upon any person in "
        "connection with the purchase or sale of any security."
    ),
    # FINRA wash trade rules
    (
        "FINRA Rule 6140(b) – Fictitious Transactions: No member shall, for the purpose of "
        "creating a false or misleading appearance of active trading in a security, enter "
        "orders for the purchase and sale of such security knowing that orders of substantially "
        "the same size at substantially the same time and at substantially the same price have "
        "been or will be entered by or for the same or different parties."
    ),
    (
        "FINRA Wash Trading Policy: A wash trade occurs when an investor simultaneously sells "
        "and buys the same financial instruments to create the appearance of active trading "
        "without any real change in beneficial ownership. Such activity violates FINRA rules "
        "and may constitute market manipulation under federal securities laws."
    ),
    (
        "FINRA Rule 4511 – Books and Records: Members must preserve trade records for at least "
        "six years. Unusual trading patterns — including high-frequency same-symbol transactions "
        "by a single trader — must be flagged and reviewed for potential wash trading or layering "
        "activity under FINRA supervision obligations."
    ),
]


# ── Index builders ────────────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokeniser."""
    return text.lower().split()


def build_bm25_index(docs: list[str]) -> BM25Okapi:
    """Build a BM25Okapi index over *docs*.

    Args:
        docs: List of document strings.

    Returns:
        Fitted :class:`BM25Okapi` instance.
    """
    tokenized = [_tokenize(doc) for doc in docs]
    return BM25Okapi(tokenized)


def build_faiss_index(
    docs: list[str],
    embeddings: OllamaEmbeddings,
) -> tuple[faiss.IndexFlatIP, np.ndarray]:
    """Embed *docs* with nomic-embed-text and build an inner-product FAISS index.

    Vectors are L2-normalised so inner product equals cosine similarity.

    Args:
        docs: List of document strings to embed.
        embeddings: :class:`OllamaEmbeddings` instance configured for nomic-embed-text.

    Returns:
        Tuple of (FAISS index, normalised embedding matrix).
    """
    vecs = np.array(embeddings.embed_documents(docs), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_norm = vecs / np.maximum(norms, 1e-10)

    index = faiss.IndexFlatIP(vecs_norm.shape[1])
    index.add(vecs_norm)
    return index, vecs_norm


# ── Retrieval ─────────────────────────────────────────────────────────────────


def hybrid_retrieve(
    query: str,
    bm25: BM25Okapi,
    faiss_index: faiss.IndexFlatIP,
    embeddings: OllamaEmbeddings,
    docs: list[str],
    top_k: int = 3,
    bm25_weight: float = 0.5,
) -> list[str]:
    """Retrieve the top-*k* documents using hybrid BM25 + cosine similarity.

    Scores from both retrievers are min-max normalised to [0, 1] then
    combined as ``bm25_weight * bm25_score + (1 - bm25_weight) * cos_score``.

    Args:
        query: Free-text search query (e.g. trade summary).
        bm25: Fitted BM25 index over the corpus.
        faiss_index: FAISS inner-product index built from normalised embeddings.
        embeddings: Embedding model used to encode the query.
        docs: Original document strings corresponding to index positions.
        top_k: Number of documents to return.
        bm25_weight: Weight given to BM25 scores (cosine gets ``1 - bm25_weight``).

    Returns:
        List of up to *top_k* document strings ranked by combined score.
    """
    # BM25 scores — normalise to [0, 1]
    bm25_scores = np.array(bm25.get_scores(_tokenize(query)), dtype=np.float32)
    bm25_max = bm25_scores.max()
    if bm25_max > 0:
        bm25_scores /= bm25_max

    # Cosine scores via FAISS
    q_vec = np.array(embeddings.embed_query(query), dtype=np.float32).reshape(1, -1)
    q_norm = q_vec / np.maximum(np.linalg.norm(q_vec), 1e-10)
    cos_raw, _ = faiss_index.search(q_norm, len(docs))
    cos_scores = cos_raw[0]

    # Combine and rank
    combined = bm25_weight * bm25_scores + (1.0 - bm25_weight) * cos_scores
    top_indices = np.argsort(combined)[::-1][:top_k]
    return [docs[i] for i in top_indices]
