# Chatbot with RAG (Retrieval Augmented Generation)

## Overview
An intelligent chatbot that uses RAG (Retrieval Augmented Generation) to answer questions based on uploaded documents. Supports both OpenAI and Ollama for LLM inference.

## Features
- Document upload and processing (PDF, DOCX, TXT)
- Vector embeddings and semantic search
- Conversational memory
- Support for OpenAI GPT-3.5/GPT-4 and Ollama (local)
- Streamlit frontend with beautiful UI
- FastAPI backend

## Tech Stack
- Backend: FastAPI
- Frontend: Streamlit
- LLM: OpenAI API / Ollama
- Embeddings: OpenAI / HuggingFace
- Vector DB: ChromaDB
- Document Processing: LangChain, pypdf, python-docx

