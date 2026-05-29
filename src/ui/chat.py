"""
Chat UI — render history, handle input, and stream Gemini responses.
"""

import streamlit as st

from src.config import COMPANY_NAME
from src.retrieval.retriever import retrieve
from src.llm.prompts import build_prompt
from src.llm.gemini_client import stream_response


def render_chat_history() -> None:
    """Display all messages stored in session state."""
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def handle_user_input(index, chunks: list[str]) -> None:
    """
    Process new user input: retrieve context, call Gemini, stream response.

    Args:
        index: The FAISS vector index.
        chunks: The list of text chunks corresponding to the index.
    """
    user_query = st.chat_input("Ask me anything about our company...")

    if not user_query:
        return

    # ── Append & display user message ────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # ── Retrieve relevant chunks ─────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            results = retrieve(user_query, index, chunks)

        from src.config import SUPPORT_PHONE, SUPPORT_EMAIL, SUPPORT_WEBSITE

        if not results:
            fallback = (
                "I'm sorry, I don't have information about that. Please contact our support team.\n\n---\n"
                "**Contact Information**  \n"
                f"Website: {SUPPORT_WEBSITE} | Email: {SUPPORT_EMAIL} | Phone: {SUPPORT_PHONE}"
            )
            st.markdown(fallback)
            st.session_state.messages.append({"role": "assistant", "content": fallback})
            return

        # ── Build prompt & stream response ───────────────────────────
        context_texts = [chunk_text for chunk_text, _score in results]
        prompt = build_prompt(COMPANY_NAME, context_texts, user_query)

        # Stream the Gemini response directly
        full_response = st.write_stream(stream_response(prompt))

        # Only append contact info if response is the fallback "unknown" response
        if "don't have information" in full_response or "contact our support team" in full_response:
            contact_info = (
                f"\n\n---\n"
                f"**Contact Information**  \n"
                f"Website: {SUPPORT_WEBSITE} | Email: {SUPPORT_EMAIL} | Phone: {SUPPORT_PHONE}"
            )
            st.markdown(contact_info)
            full_response += contact_info

        st.session_state.messages.append({"role": "assistant", "content": full_response})



