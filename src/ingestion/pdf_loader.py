"""
PDF text extraction using pdfplumber.
"""

from pathlib import Path


def load_pdf(pdf_path: str) -> str:
    """
    Extract full text from a PDF file, page by page.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Concatenated text from all pages.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If text extraction fails.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages_text: list[str] = []

    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    pages_text.append(text.strip())
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}") from e

    if not pages_text:
        raise RuntimeError(f"No text could be extracted from {path}")

    return "\n\n".join(pages_text)
