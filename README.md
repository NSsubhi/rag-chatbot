# Chatbot with RAG (Retrieval Augmented Generation)

## Overview
An intelligent chatbot that uses RAG (Retrieval Augmented Generation) to answer questions based on uploaded documents. Supports OpenAI, HuggingFace (free), and Ollama for LLM inference.

## Features
- Document upload and processing (PDF, DOCX, TXT)
- Vector embeddings and semantic search
- Conversational memory
- Support for OpenAI GPT-3.5/GPT-4, HuggingFace (free), and Ollama (local)
- Streamlit frontend with beautiful UI
- FastAPI backend

## Tech Stack
- Backend: FastAPI
- Frontend: Streamlit
- LLM: OpenAI API / HuggingFace (free) / Ollama (local)
- Embeddings: OpenAI / HuggingFace (free)
- Vector DB: ChromaDB
- Document Processing: LangChain, pypdf, python-docx

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
```

Edit `.env` and configure your LLM (optional - will use free HuggingFace models if not provided):

**Option 1: OpenAI (requires API key)**
```
OPENAI_API_KEY=your_api_key_here
MODEL_NAME=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002
```

**Option 2: HuggingFace (free - no API key needed for local models)**
```
# Leave OPENAI_API_KEY empty or comment it out
# The system will automatically use HuggingFace models
```

**Option 3: HuggingFace Inference API (free tier)**
```
HUGGINGFACE_API_TOKEN=your_hf_token_here
HUGGINGFACE_MODEL=google/flan-t5-base
```

**Option 4: Ollama (requires local installation)**
```
OLLAMA_MODEL=llama2
```

### 3. Run the Application

**Terminal 1 - Start Backend (FastAPI):**
```bash
python -m app.main
# or
uvicorn app.main:app --reload
```

**Terminal 2 - Start Frontend (Streamlit):**
```bash
streamlit run app/frontend.py
```

The application will be available at:
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000

### 4. Usage
1. Open the Streamlit frontend in your browser
2. Upload a document (PDF, DOCX, or TXT) using the sidebar
3. Ask questions about the uploaded document
4. The chatbot will use RAG to answer based on the document content

## Notes
- **Free LLM Option**: If OpenAI API key is not provided, the system will automatically use free HuggingFace models (local inference, no API needed)
- Documents are processed and stored in a ChromaDB vector database
- The vector database persists in `./data/chroma/` directory (configurable via `CHROMA_DB_PATH`)
- HuggingFace models will download on first use (requires internet connection)
- For best performance, consider using OpenAI API or Ollama with local models

