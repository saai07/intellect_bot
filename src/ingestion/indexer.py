"""
FAISS index builder — build, save, and load the vector store.
"""

import pickle
from pathlib import Path

import numpy as np

from src.config import FAISS_INDEX_PATH, CHUNKS_PKL_PATH, VECTOR_STORE_DIR


def build_index(embeddings: np.ndarray):
    """
    Build a FAISS inner-product index from L2-normalised embeddings
    (equivalent to cosine similarity search).

    Args:
        embeddings: 2-D float32 array of shape (n, dim).

    Returns:
        Populated FAISS IndexFlatIP.
    """
    import faiss
    # L2-normalise so inner product == cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index, chunks: list[str]) -> None:
    """Persist FAISS index and chunk texts to disk."""
    import faiss
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    with open(CHUNKS_PKL_PATH, "wb") as f:
        pickle.dump(chunks, f)


def load_index() -> tuple or None:
    """
    Load a previously saved FAISS index and chunk list.

    Returns:
        (index, chunks) tuple, or None if files don't exist.
    """
    if not FAISS_INDEX_PATH.exists() or not CHUNKS_PKL_PATH.exists():
        return None

    import faiss
    index = faiss.read_index(str(FAISS_INDEX_PATH))

    with open(CHUNKS_PKL_PATH, "rb") as f:
        chunks: list[str] = pickle.load(f)

    return index, chunks

