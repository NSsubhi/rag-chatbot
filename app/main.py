"""
FastAPI Backend for RAG Chatbot
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
from dotenv import load_dotenv

try:
    from .rag_engine import RAGEngine
    from .document_processor import DocumentProcessor
except ImportError:
    from rag_engine import RAGEngine
    from document_processor import DocumentProcessor

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval Augmented Generation Chatbot API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
rag_engine = RAGEngine()
document_processor = DocumentProcessor()

# Request models
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    use_rag: bool = True

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = None
    conversation_id: Optional[str] = None

# Routes
@app.get("/")
async def root():
    return {"message": "RAG Chatbot API", "status": "running"}

@app.get("/health")
async def health():
    status = {
        "status": "healthy",
        "rag_ready": rag_engine.is_ready(),
        "llm_configured": rag_engine.llm is not None,
        "embeddings_configured": rag_engine.embeddings is not None
    }
    return status

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not rag_engine.is_ready():
            return ChatResponse(
                response="Please configure an LLM (OpenAI or Ollama) for full RAG functionality.",
                sources=None
            )
        
        response, sources = rag_engine.generate_response(
            request.message,
            conversation_id=request.conversation_id,
            use_rag=request.use_rag
        )
        
        return ChatResponse(
            response=response,
            sources=sources,
            conversation_id=request.conversation_id
        )
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-document")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Read file content
        content = await file.read()
        
        # Process document
        chunks = document_processor.process_file(file.filename, content)
        
        # Add to vector store
        if rag_engine.is_ready():
            rag_engine.add_documents(chunks)
            return {
                "status": "success",
                "message": f"Document processed and added to knowledge base. {len(chunks)} chunks created.",
                "chunks": len(chunks)
            }
        else:
            return {
                "status": "warning",
                "message": "Document chunks prepared but not added to vector DB. Please configure LLM first.",
                "chunks": len(chunks)
            }
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/knowledge-base/status")
async def get_knowledge_base_status():
    return {
        "ready": rag_engine.is_ready(),
        "vector_store_ready": rag_engine.vector_store is not None,
        "documents_count": len(rag_engine.vector_store.get()['ids']) if rag_engine.vector_store else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

