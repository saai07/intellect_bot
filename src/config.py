"""
Application configuration — loads environment variables and defines constants.
"""

import os
from pathlib import Path
import warnings
import logging

# Suppress deprecation and user warnings globally
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

from dotenv import load_dotenv

# ── Load .env from project root ──────────────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── API & Identity ───────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
COMPANY_NAME: str = os.getenv("COMPANY_NAME", "MyCompany")
SUPPORT_EMAIL: str = os.getenv("SUPPORT_EMAIL", "sales@saintellectsolutions.com")
SUPPORT_PHONE: str = os.getenv("SUPPORT_PHONE", "+91 7077211601")
SUPPORT_WEBSITE: str = os.getenv("SUPPORT_WEBSITE", "saintellectsolutions.com")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Read path from env, defaulting to data/company_knowledge.pdf
env_pdf_path = os.getenv("PDF_PATH", "data/company_knowledge.pdf")

# If it is a relative path (e.g. starts with ./ or data/), resolve it relative to PROJECT_ROOT
pdf_path_obj = Path(env_pdf_path)
if not pdf_path_obj.is_absolute():
    pdf_path_obj = PROJECT_ROOT / pdf_path_obj

PDF_PATH: str = str(pdf_path_obj.resolve())

VECTOR_STORE_DIR: Path = PROJECT_ROOT / "vector_store"
FAISS_INDEX_PATH: Path = VECTOR_STORE_DIR / "faiss.index"
CHUNKS_PKL_PATH: Path = VECTOR_STORE_DIR / "chunks.pkl"

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

# ── Embedding ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K: int = 5

# ── LLM ──────────────────────────────────────────────────────────────────────
GEMINI_MODEL: str = "gemini-2.5-flash"
