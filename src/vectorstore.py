"""
vectorstore.py
--------------
Loads the saved FAISS index from disk.
Called by retriever.py and chain.py — not run directly.
"""

from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTOR_STORE_DIR = Path("vector_store")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_vector_store():
    """Load FAISS index from disk. Returns a FAISS vector store object."""
    if not (VECTOR_STORE_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            "Vector store not found. Run: python src/ingest.py"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store
