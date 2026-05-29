"""
Retriever — embed a user query and fetch top-k relevant chunks from FAISS.
"""

import numpy as np

from src.config import TOP_K
from src.ingestion.embedder import embed_query



def retrieve(
    query: str,
    index,
    chunks: list[str],
    top_k: int = TOP_K,
) -> list[tuple[str, float]]:
    """
    Retrieve the most relevant chunks for a given query.

    Args:
        query: User's natural-language question.
        index: Populated FAISS index.
        chunks: Ordered list of chunk texts corresponding to index IDs.
        top_k: Number of results to return.

    Returns:
        List of (chunk_text, similarity_score) tuples, highest score first.
    """
    import faiss
    if not query or not query.strip():
        return []

    query_vec = embed_query(query).reshape(1, -1).astype(np.float32)

    # L2-normalise the query vector to match the indexed vectors
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, min(top_k, index.ntotal))

    results: list[tuple[str, float]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append((chunks[idx], float(score)))

    return results
