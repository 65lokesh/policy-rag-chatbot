"""
retriever.py
------------
Wraps the FAISS vector store with MMR retrieval.

MMR (Maximal Marginal Relevance) avoids returning 5 near-identical chunks.
It balances relevance AND diversity — key to the "35% context relevance" metric.
"""

from src.vectorstore import load_vector_store

# How many chunks to retrieve per query
TOP_K = 5
# MMR lambda: 0 = max diversity, 1 = max relevance. 0.7 is a good balance.
MMR_LAMBDA = 0.7
# Candidate pool before MMR re-ranks (fetch_k > k)
FETCH_K = 20


def get_retriever():
    """Return a LangChain retriever using MMR search."""
    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K,
            "fetch_k": FETCH_K,
            "lambda_mult": MMR_LAMBDA,
        },
    )
    return retriever
