"""
Streamlit sidebar — company branding, KB status, and controls.
"""

import streamlit as st

from src.config import COMPANY_NAME


def render_sidebar(kb_loaded: bool) -> None:
    """
    Render a minimal sidebar with only a Clear Chat button.
    """
    with st.sidebar:
        st.write("")  # Padding spacer
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


