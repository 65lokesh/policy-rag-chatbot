"""
ingest.py
---------
Loads all PDFs from the data/ folder, splits them into chunks,
embeds them using sentence-transformers (free, runs locally),
and saves the FAISS vector store to vector_store/.

Run this ONCE before starting the chatbot:
    python src/ingest.py
"""

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = Path("data")
VECTOR_STORE_DIR = Path("vector_store")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # free, runs on CPU, ~80MB download
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def load_documents():
    """Load all PDFs from the data/ folder."""
    all_docs = []
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in data/. Please add your policy PDFs there.")
        return []

    print(f"Found {len(pdf_files)} PDF(s). Loading...")

    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = pdf_path.name
        all_docs.extend(docs)

    print(f"  Total pages loaded: {len(all_docs)}")
    return all_docs


def chunk_documents(docs):
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    total_tokens = sum(len(c.page_content.split()) for c in chunks)
    print(f"  Created {len(chunks)} chunks (~{total_tokens:,} tokens)")
    return chunks


def build_vector_store(chunks):
    """Embed chunks and save FAISS index to disk."""
    print(f"  Loading embedding model: {EMBEDDING_MODEL} (first run downloads ~80MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("  Building FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    VECTOR_STORE_DIR.mkdir(exist_ok=True)
    vector_store.save_local(str(VECTOR_STORE_DIR))
    print(f"  Vector store saved to {VECTOR_STORE_DIR}/")
    return vector_store


if __name__ == "__main__":
    print("=== RAG Ingestion Pipeline ===\n")

    docs = load_documents()
    if not docs:
        exit(1)

    chunks = chunk_documents(docs)
    build_vector_store(chunks)

    print("\nDone! You can now run: streamlit run app.py")
