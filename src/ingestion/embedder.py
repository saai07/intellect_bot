import numpy as np

from src.config import EMBEDDING_MODEL

# Module-level cache so the model is loaded only once
_model = None



def _get_model():
    """Lazily load and cache the sentence-transformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Encode a list of texts into dense embeddings.

    Args:
        texts: List of text strings to embed.

    Returns:
        2-D numpy array of shape (len(texts), embedding_dim).
    """
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return np.array(embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Encode a single query string.

    Args:
        query: The query text.

    Returns:
        1-D numpy array of shape (embedding_dim,).
    """
    model = _get_model()
    embedding = model.encode([query], convert_to_numpy=True)
    return np.array(embedding, dtype=np.float32).squeeze()
