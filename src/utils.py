"""
utils.py
--------
Helper functions used across the project.
"""


def format_sources(source_docs: list) -> str:
    """
    Format source documents into a readable citation string for the UI.
    Shows which PDF file and page number each answer came from.
    """
    if not source_docs:
        return ""

    seen = set()
    citations = []

    for doc in source_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        key = f"{source}-p{page}"

        if key not in seen:
            seen.add(key)
            citations.append(f"- {source} (page {page})")

    return "\n".join(citations)


def count_tokens_approx(text: str) -> int:
    """Rough token estimate: 1 token ≈ 0.75 words."""
    return int(len(text.split()) / 0.75)


def clean_text(text: str) -> str:
    """
    Basic text cleaning for PDF content.
    Removes excessive whitespace and common PDF artifacts.
    """
    import re

    # Remove excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove page numbers standing alone on a line
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()
