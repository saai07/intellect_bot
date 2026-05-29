"""
Google Gemini client — configure SDK, stream responses.
Uses the `google-genai` SDK (Client-based API).
"""

from collections.abc import Generator

from google import genai

from src.config import GOOGLE_API_KEY, GEMINI_MODEL

# Create a client instance once at import time
_client = genai.Client(api_key=GOOGLE_API_KEY)


def stream_response(prompt: str) -> Generator[str, None, None]:
    """
    Send a prompt to Gemini and yield response tokens as they arrive.

    Args:
        prompt: The fully assembled prompt string.

    Yields:
        Text chunks from the streaming response.

    Raises:
        RuntimeError: On API errors (caught by caller for UI display).
    """
    try:
        response = _client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    except Exception as e:
        yield f"\n\n**Error communicating with Gemini:** {e}"


