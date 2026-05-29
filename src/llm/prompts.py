"""
Prompt templates for the company chatbot.
"""

SYSTEM_PROMPT_TEMPLATE = """You are a helpful customer support agent for {company_name}.
Answer the customer's question using ONLY the context provided below.
If the answer is not in the context, say: "I'm sorry, I don't have information about that. Please contact our support team."
Do not make up information. Be concise and friendly."""


def build_prompt(company_name: str, chunks: list[str], user_query: str) -> str:
    """
    Assemble the full prompt with system instructions, retrieved context,
    and the user's question.

    Args:
        company_name: Name of the company for personalisation.
        chunks: Retrieved context chunks.
        user_query: The customer's question.

    Returns:
        Formatted prompt string.
    """
    system = SYSTEM_PROMPT_TEMPLATE.format(company_name=company_name)

    context_block = "\n\n".join(
        f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )

    prompt = (
        f"{system}\n\n"
        f"--- CONTEXT ---\n"
        f"{context_block}\n"
        f"--- END CONTEXT ---\n\n"
        f"Customer question: {user_query}"
    )

    return prompt
