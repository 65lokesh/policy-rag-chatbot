"""
prompts.py
----------
System prompt templates for the RAG chatbot.

The prompt is the main hallucination guard:
- Instructs the LLM to ONLY use retrieved context
- Adds a fallback when context is insufficient
- Includes source citation instruction
"""

from langchain.prompts import PromptTemplate

# Main RAG prompt — used inside the ConversationalRetrievalChain
RAG_PROMPT_TEMPLATE = """You are a helpful policy assistant. Your job is to answer questions
about company policies clearly and accurately.

STRICT RULES:
1. Answer ONLY using the context provided below.
2. If the answer is not in the context, say: "I don't have enough information in the loaded policy documents to answer that."
3. Always mention which document your answer came from (e.g., "According to openai_privacy_policy.pdf...").
4. Be concise and clear. Use bullet points for lists.

Context from policy documents:
{context}

Chat history:
{chat_history}

Question: {question}

Answer:"""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=RAG_PROMPT_TEMPLATE,
)

# Condense question prompt — rewrites follow-up questions using chat history
# e.g. "What about children?" → "What does OpenAI's policy say about children's data?"
CONDENSE_QUESTION_TEMPLATE = """Given the following conversation and a follow-up question,
rephrase the follow-up question to be a standalone question that includes all necessary context.

Chat history:
{chat_history}

Follow-up question: {question}

Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=CONDENSE_QUESTION_TEMPLATE,
)
