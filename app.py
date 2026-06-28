"""
app.py
------
Streamlit UI for the Policy RAG Chatbot.

Run:
    streamlit run app.py
"""

import streamlit as st
from src.chain import build_chain
from src.utils import format_sources

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Policy RAG Chatbot",
    page_icon="📄",
    layout="centered",
)

st.title("📄 Policy RAG Chatbot")
st.caption("Ask questions about loaded company policy documents.")

# ── Load chain (once per session) ────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def get_chain():
    return build_chain()

try:
    chain = get_chain()
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Run this command first:  `python src/ingest.py`")
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()

# ── Chat history in session state ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render existing messages ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                st.markdown(msg["sources"])

# ── Chat input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask about the policy documents..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get RAG response
    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents..."):
            result = chain.invoke({"question": user_input})
            answer = result["answer"]
            sources = format_sources(result.get("source_documents", []))

        st.markdown(answer)

        if sources:
            with st.expander("Sources"):
                st.markdown(sources)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown("""
    **How it works:**
    1. PDFs in `data/` are chunked & embedded
    2. Your question is embedded the same way
    3. MMR retrieval finds the top-5 relevant chunks
    4. Gemini Flash answers using only those chunks
    5. Sources are shown so you can verify

    **To reset chat:**
    """)
    if st.button("Clear conversation"):
        st.session_state.messages = []
        # Also reset chain memory
        chain.memory.clear()
        st.rerun()

    st.divider()
    st.markdown("**Sample questions:**")
    sample_questions = [
        "What data does OpenAI collect?",
        "How long is data retained?",
        "Can I delete my data?",
        "How are cookies used?",
        "What are my GDPR rights?",
    ]
    for q in sample_questions:
        if st.button(q, key=q):
            st.session_state["_inject_question"] = q
            st.rerun()
