"""
IntellectBot — Company Chatbot
Streamlit entry point that wires together ingestion, retrieval, and LLM layers.
"""

import streamlit as st
import warnings
import logging

# Silence deprecation and user warnings from packages
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

from src.config import PDF_PATH, COMPANY_NAME
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import split_text
from src.ingestion.embedder import embed_texts
from src.ingestion.indexer import build_index, save_index, load_index
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.ui.sidebar import render_sidebar
from src.ui.chat import render_chat_history, handle_user_input


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"{COMPANY_NAME} Support",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS for a polished look ───────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Main container ─────────────────────────────────────────── */
    .stApp {
        background-color: #0A0A0B;
    }

    /* ── Chat messages ──────────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        border-radius: 8px;
        margin-bottom: 0.75rem;
        padding: 1rem 1.25rem;
        background-color: #121214 !important;
        border: 1px solid #1F1F23;
    }

    /* ── User message ───────────────────────────────────────────── */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        border-left: 3px solid #00F0FF;
    }

    /* ── Assistant message ──────────────────────────────────────── */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        border-left: 3px solid #A855F7;
    }

    /* ── Sidebar ────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #121214;
        border-right: 1px solid #1F1F23;
    }

    /* ── Chat input ─────────────────────────────────────────────── */
    [data-testid="stChatInput"] textarea {
        background-color: #121214 !important;
        border: 1px solid #1F1F23 !important;
        border-radius: 8px !important;
        color: #E2E8F0 !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #00F0FF !important;
    }

    /* ── Buttons ────────────────────────────────────────────────── */
    .stButton > button {
        background-color: #121214;
        color: #E2E8F0;
        border: 1px solid #1F1F23;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #00F0FF;
        color: #00F0FF;
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.2);
    }

    /* ── Headings ───────────────────────────────────────────────── */
    h1, h2, h3 {
        color: #F8F9FA !important;
    }

    /* ── Divider ────────────────────────────────────────────────── */
    hr {
        border-color: #1F1F23 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Knowledge base loader (cached — runs once) ──────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_knowledge_base():
    """
    Ingest PDF → chunk → embed → build FAISS index.
    Cached so it only runs once per Streamlit server lifetime.
    """
    # Try loading a persisted index first
    cached = load_index()
    if cached is not None:
        return cached

    # Full ingestion pipeline
    text = load_pdf(PDF_PATH)
    chunks = split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    if not chunks:
        raise RuntimeError("No chunks produced from PDF")

    embeddings = embed_texts(chunks)
    index = build_index(embeddings)

    # Persist for future runs
    save_index(index, chunks)

    return index, chunks


# ── Initialise session state ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Load KB & render UI ──────────────────────────────────────────────────────
kb_loaded = False
index = None
chunks = []

try:
    index, chunks = load_knowledge_base()
    kb_loaded = True
except Exception as e:
    st.error(f"Error loading knowledge base: {e}")

# Sidebar
render_sidebar(kb_loaded)

# Header
st.markdown(
    f"""
    <div style="text-align:center; padding: 2rem 0 1rem;">
        <h1 style="
            font-size: 2.2rem;
            color: #F8F9FA;
            margin-bottom: 0.25rem;
        ">{COMPANY_NAME}</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Chat
render_chat_history()

if kb_loaded and index is not None:
    handle_user_input(index, chunks)

