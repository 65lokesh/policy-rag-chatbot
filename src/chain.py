"""
chain.py
--------
Builds the full ConversationalRetrievalChain using:
  - Groq LLaMA (free LLM)
  - FAISS retriever (MMR)
  - Conversation memory (multi-turn chat)
  - Custom prompts (hallucination guard)
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from src.retriever import get_retriever
from src.prompts import RAG_PROMPT, CONDENSE_QUESTION_PROMPT

load_dotenv()


def build_chain():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Add to your .env file:\n"
            "GROQ_API_KEY=your_key_here\n"
            "Get your free key at: https://console.groq.com"
        )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0.1,
        max_tokens=1024,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    retriever = get_retriever()

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": RAG_PROMPT},
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        return_source_documents=True,
        verbose=False,
    )

    return chain