#!/bin/bash
# Run this once to create the full project structure
# Usage: bash setup.sh

mkdir -p policy-rag-chatbot/data
mkdir -p policy-rag-chatbot/src
mkdir -p policy-rag-chatbot/vector_store

touch policy-rag-chatbot/data/.gitkeep
touch policy-rag-chatbot/vector_store/.gitkeep

echo "✅ Folder structure created!"
echo ""
echo "Next steps:"
echo "  1. cd policy-rag-chatbot"
echo "  2. Add your PDFs inside data/"
echo "  3. Copy .env.example to .env and add your Gemini API key"
echo "  4. pip install -r requirements.txt"
echo "  5. python src/ingest.py        (run once to build vector store)"
echo "  6. streamlit run app.py        (start the chatbot)"
