# Policy RAG Chatbot

An LLM-powered RAG chatbot that answers questions about company policy documents.

## Stack
- **LLM**: Gemini 1.5 Flash (free API)
- **Embeddings**: sentence-transformers all-MiniLM-L6-v2 (local, free)
- **Vector Store**: FAISS
- **Framework**: LangChain
- **UI**: Streamlit

## Setup

### 1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get free Gemini API key
Go to https://aistudio.google.com → Get API Key

### 4. Add API key
```bash
cp .env.example .env
# Edit .env and paste your key
```

### 5. Add policy PDFs
- Open https://openai.com/policies/privacy-policy in Chrome
- Press Ctrl+P → Save as PDF
- Save to the `data/` folder

### 6. Build vector store (run once)
```bash
python src/ingest.py
```

### 7. Start the chatbot
```bash
streamlit run app.py
```

## Running the benchmark
```bash
python src/evaluate.py
```

## Project Structure
```
policy-rag-chatbot/
├── data/                  ← Put your PDFs here
├── src/
│   ├── ingest.py          ← Load + chunk + embed PDFs
│   ├── vectorstore.py     ← Load FAISS index
│   ├── retriever.py       ← MMR retrieval
│   ├── prompts.py         ← System prompt (hallucination guard)
│   ├── chain.py           ← LangChain RAG chain
│   ├── evaluate.py        ← QA benchmark script
│   └── utils.py           ← Helpers
├── vector_store/          ← Auto-generated FAISS index
├── app.py                 ← Streamlit UI
├── requirements.txt
└── .env                   ← Your API key (never commit this)
```
