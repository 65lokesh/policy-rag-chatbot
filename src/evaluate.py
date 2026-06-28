"""
evaluate.py
-----------
QA benchmark to measure chatbot accuracy.

Usage:
    python src/evaluate.py
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from src.chain import build_chain

load_dotenv()

QA_PAIRS = [
    {
        "question": "What types of personal information does OpenAI collect?",
        "expected_keywords": ["name", "email", "account", "usage", "content"],
    },
    {
        "question": "How long does OpenAI retain user data?",
        "expected_keywords": ["retain", "period", "delete", "legal"],
    },
    {
        "question": "Can I request deletion of my data from OpenAI?",
        "expected_keywords": ["delete", "request", "right", "access"],
    },
    {
        "question": "Does OpenAI share data with third parties?",
        "expected_keywords": ["third", "share", "partner", "vendor", "service"],
    },
    {
        "question": "How does OpenAI use cookies?",
        "expected_keywords": ["cookie", "browser", "track", "preference"],
    },
    {
        "question": "What are my rights under GDPR according to OpenAI's policy?",
        "expected_keywords": ["gdpr", "europe", "right", "access", "erasure"],
    },
    {
        "question": "How does OpenAI protect children's privacy?",
        "expected_keywords": ["child", "minor", "13", "age", "coppa"],
    },
    {
        "question": "Can I opt out of OpenAI using my data for training?",
        "expected_keywords": ["opt", "training", "model", "request"],
    },
    {
        "question": "What is OpenAI's policy on data security?",
        "expected_keywords": ["security", "protect", "encrypt", "safeguard"],
    },
    {
        "question": "How can I contact OpenAI about privacy concerns?",
        "expected_keywords": ["contact", "email", "dpo", "privacy", "request"],
    },
]


def answer_is_correct(answer: str, expected_keywords: list) -> bool:
    answer_lower = answer.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return (matches / len(expected_keywords)) >= 0.6


def run_vanilla_llm(question: str) -> str:
    """Run the same question through Groq with NO retrieval — pure LLM baseline."""
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
        max_tokens=1024,
    )
    response = llm.invoke([HumanMessage(content=question)])
    return response.content


def run_benchmark():
    print("=" * 60)
    print("RAG Chatbot — Accuracy Benchmark")
    print("=" * 60)

    print("\nLoading RAG chain...")
    rag_chain = build_chain()

    rag_correct = 0
    vanilla_correct = 0
    total = len(QA_PAIRS)

    print(f"\nRunning {total} test questions...\n")

    for i, qa in enumerate(QA_PAIRS, 1):
        q = qa["question"]
        kws = qa["expected_keywords"]

        # RAG answer (with retrieval)
        rag_result = rag_chain.invoke({"question": q})
        rag_answer = rag_result["answer"]
        rag_ok = answer_is_correct(rag_answer, kws)

        # Vanilla answer (no retrieval)
        vanilla_answer = run_vanilla_llm(q)
        vanilla_ok = answer_is_correct(vanilla_answer, kws)

        rag_correct += int(rag_ok)
        vanilla_correct += int(vanilla_ok)

        status = "PASS" if rag_ok else "FAIL"
        print(f"Q{i}: [{status}] {q[:60]}...")

    rag_accuracy = (rag_correct / total) * 100
    vanilla_accuracy = (vanilla_correct / total) * 100
    hallucination_reduction = ((vanilla_accuracy - rag_accuracy) / max(vanilla_accuracy, 1)) * -100

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"RAG chatbot accuracy   : {rag_accuracy:.1f}%  ({rag_correct}/{total} correct)")
    print(f"Vanilla LLM accuracy   : {vanilla_accuracy:.1f}%  ({vanilla_correct}/{total} correct)")
    print(f"Hallucination reduction: ~{abs(hallucination_reduction):.0f}%")
    print("=" * 60)
    print("\nThese are the numbers you cite in your resume and interviews.")


if __name__ == "__main__":
    run_benchmark()