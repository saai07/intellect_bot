# Intellect Support Bot — AI Company Chatbot

A RAG-powered customer support chatbot built with Streamlit, Google Gemini, and FAISS vector search. The bot answers customer queries using your company's knowledge base extracted from a PDF.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![Gemini](https://img.shields.io/badge/LLM-Gemini%202.5-green)

---

## Features

- **PDF Knowledge Base** — Automatically extracts and indexes content from your company PDF
- **Semantic Search** — FAISS + sentence-transformers for accurate context retrieval
- **Gemini LLM** — Google's Gemini 2.5 Flash for fast, grounded responses
- **Streaming Responses** — Token-by-token streaming for a real-time chat experience
- **Native Chat UI** — Streamlit's built-in chat components with full history
- **Premium Dark Theme** — Polished minimal dark UI with sharp neon cyan and purple borders
- **Conditional Contact Card** — Displays support contact information only when an answer is not found in the knowledge base

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────┐
│  Embed Query     │  ← sentence-transformers (all-MiniLM-L6-v2)
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  FAISS Search    │  ← Cosine similarity, top-5 chunks
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Prompt Builder  │  ← System prompt + retrieved context + user query
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Gemini API      │  ← Streaming response
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Streamlit Chat  │  ← Token-by-token display
└──────────────────┘
```

---

## Project Structure

```
intellect_bot/
├── app.py                          # Streamlit entry point
├── .env                            # Environment variables
├── requirements.txt                # Python dependencies
├── README.md
│
├── src/
│   ├── config.py                   # Configuration & constants
│   ├── ingestion/
│   │   ├── pdf_loader.py           # PDF text extraction (pdfplumber)
│   │   ├── chunker.py              # Recursive text splitter
│   │   ├── embedder.py             # Sentence-transformer embeddings
│   │   └── indexer.py              # FAISS index build/save/load
│   ├── retrieval/
│   │   └── retriever.py            # Query → top-k chunk retrieval
│   ├── llm/
│   │   ├── prompts.py              # System prompt template
│   │   └── gemini_client.py        # Gemini SDK streaming client
│   └── ui/
│       ├── sidebar.py              # Sidebar: clear button
│       └── chat.py                 # Chat history, stream & fallback handler
│
├── data/
│   └── your_Knowledge_Base.pdf  # Company PDF
│
└── vector_store/                   # Auto-created on first run
    ├── faiss.index
    └── chunks.pkl
```

---

## Setup & Run

### 1. Clone the repository

```bash
cd intellect_bot
```

### 2. Create a Conda environment

```bash
conda create -n intellect_bot python=3.11 -y
conda activate intellect_bot
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install "numpy<2"
```

### 4. Place your PDF

Place your company knowledge base PDF inside the `data` folder.

### 5. Configure environment variables

Edit the `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key_here
COMPANY_NAME=your_company_name_here
PDF_PATH=./data/your_company_knowledge.pdf
SUPPORT_EMAIL=support@yourdomain.com
SUPPORT_PHONE=+1 (555) 019-2834
SUPPORT_WEBSITE=yourdomain.com
```

> Get your Gemini API key from Google AI Studio (https://aistudio.google.com/apikey)

### 6. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Configuration

All settings are defined in `src/config.py` and can be overridden via `.env`:

| Setting           | Default                           | Description                    |
| ----------------- | --------------------------------- | ------------------------------ |
| `GOOGLE_API_KEY`  | —                                 | Your Gemini API key            |
| `COMPANY_NAME`    | `your_company_name_here`          | Displayed in UI & prompts      |
| `PDF_PATH`        | `data/your_company_knowledge.pdf` | Path to your knowledge base    |
| `SUPPORT_EMAIL`   | `support@yourdomain.com`          | Support email                  |
| `SUPPORT_PHONE`   | `+1 (555) 019-2834`               | Support phone                  |
| `SUPPORT_WEBSITE` | `yourdomain.com`                  | Support website                |
| `CHUNK_SIZE`      | `500`                             | Max characters per chunk       |
| `CHUNK_OVERLAP`   | `50`                              | Overlap between chunks         |
| `TOP_K`           | `5`                               | Number of chunks to retrieve   |
| `GEMINI_MODEL`    | `gemini-2.5-flash`                | Gemini model to use            |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2`               | Sentence-transformer model     |

---

## Getting a Gemini API Key

1. Visit Google AI Studio (https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and paste it in your `.env` file

---

## License

MIT
