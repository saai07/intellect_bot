"""
Recursive character text splitter with configurable chunk size and overlap.
"""


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks using a recursive character strategy.

    The splitter tries to break on paragraph boundaries first, then sentences,
    then words, and finally characters — preserving semantic coherence.

    Args:
        text: The full text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    # Separators ordered by preference (paragraph → sentence → word → char)
    separators = ["\n\n", "\n", ". ", " ", ""]

    chunks: list[str] = []
    _recursive_split(text.strip(), separators, chunk_size, chunk_overlap, chunks)
    return chunks


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
    result: list[str],
) -> None:
    """Recursively split text using the best available separator."""
    if len(text) <= chunk_size:
        if text.strip():
            result.append(text.strip())
        return

    # Pick the first separator that actually appears in the text
    separator = ""
    for sep in separators:
        if sep == "" or sep in text:
            separator = sep
            break

    if separator == "":
        # Last resort: hard-cut by character
        _hard_split(text, chunk_size, chunk_overlap, result)
        return

    parts = text.split(separator)
    current_chunk = ""

    for part in parts:
        candidate = (current_chunk + separator + part).strip() if current_chunk else part.strip()

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            # Flush current chunk
            if current_chunk.strip():
                result.append(current_chunk.strip())

            # If the single part itself exceeds chunk_size, recurse deeper
            if len(part.strip()) > chunk_size:
                remaining_seps = separators[separators.index(separator) + 1 :]
                _recursive_split(part.strip(), remaining_seps, chunk_size, chunk_overlap, result)
                current_chunk = ""
            else:
                # Start new chunk with overlap from previous chunk
                if chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-chunk_overlap:]
                    current_chunk = overlap_text + separator + part.strip()
                else:
                    current_chunk = part.strip()

    # Don't forget the last chunk
    if current_chunk.strip():
        result.append(current_chunk.strip())


def _hard_split(text: str, chunk_size: int, chunk_overlap: int, result: list[str]) -> None:
    """Fall-back: split text by raw character positions."""
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            result.append(chunk)
        start += chunk_size - chunk_overlap
