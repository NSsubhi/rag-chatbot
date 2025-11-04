"""
RAG Engine - Core RAG implementation
Handles embeddings, vector search, and LLM interaction
"""

import os
import logging
from typing import List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        self.embeddings = None
        self.llm = None
        self.vector_store = None
        self.conversation_memory = {}
        self._initialize_embeddings()
        self._initialize_llm()
        self._initialize_vector_store()
    
    def _initialize_embeddings(self):
        """Initialize embeddings model"""
        try:
            # Try OpenAI embeddings first
            try:
                from langchain_openai import OpenAIEmbeddings
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                    logger.info("Using OpenAI embeddings")
                    return
            except ImportError:
                pass
            
            # Fallback to HuggingFace
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Using HuggingFace embeddings")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
    
    def _initialize_llm(self):
        """Initialize LLM (OpenAI or Ollama)"""
        try:
            # Try OpenAI first
            try:
                from langchain_openai import ChatOpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.llm = ChatOpenAI(
                        model_name="gpt-3.5-turbo",
                        openai_api_key=api_key,
                        temperature=0.7,
                        timeout=90,
                        max_retries=2
                    )
                    logger.info("Using OpenAI ChatOpenAI")
                    return
            except ImportError:
                logger.warning("langchain-openai not available. Install: pip install langchain-openai")
            except Exception as e:
                logger.warning(f"OpenAI initialization failed: {e}")
            
            # Fallback to Ollama
            try:
                from langchain_community.llms import Ollama
                self.llm = Ollama(model="llama2")
                logger.info("Using Ollama (local)")
            except ImportError:
                logger.warning("Ollama not available. Install: pip install langchain-community ollama")
            except Exception as e:
                logger.warning(f"Ollama initialization failed: {e}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
    
    def _initialize_vector_store(self):
        """Initialize Chroma vector store"""
        try:
            import chromadb
            from langchain.vectorstores import Chroma
            
            # Initialize Chroma
            persist_directory = "./chroma_db"
            os.makedirs(persist_directory, exist_ok=True)
            
            if self.embeddings:
                self.vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info("Vector store initialized")
        except ImportError:
            logger.warning("ChromaDB not available. Install: pip install chromadb")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
    
    def is_ready(self) -> bool:
        """Check if RAG engine is ready"""
        return self.llm is not None and self.embeddings is not None and self.vector_store is not None
    
    def add_documents(self, chunks: List[str], batch_size: int = 50):
        """Add documents to vector store"""
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return
        
        try:
            # Process in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                self.vector_store.add_texts(batch)
                logger.info(f"Added batch {i//batch_size + 1} ({len(batch)} chunks)")
            
            logger.info(f"Successfully added {len(chunks)} chunks to vector store")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def generate_response(self, query: str, conversation_id: Optional[str] = None, use_rag: bool = True) -> Tuple[str, Optional[List[str]]]:
        """Generate response using RAG"""
        if not self.llm:
            return (
                "Please configure an LLM (OpenAI or Ollama) for full RAG functionality. "
                "Please restart the backend server after installing langchain-openai.",
                None
            )
        
        try:
            # Retrieve relevant chunks
            relevant_chunks = []
            sources = []
            
            if use_rag and self.vector_store:
                try:
                    docs = self.vector_store.similarity_search(query, k=3)
                    relevant_chunks = [doc.page_content for doc in docs]
                    sources = [doc.metadata.get('source', 'Unknown') for doc in docs]
                except Exception as e:
                    logger.warning(f"Vector search failed: {e}")
            
            # Build prompt
            if relevant_chunks:
                context = "\n\n".join(relevant_chunks)
                prompt = f"""Based on the following context, answer the question. If the answer is not in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
            else:
                prompt = f"Answer the following question: {query}"
            
            # Generate response
            try:
                # Try newer LangChain API (0.1.x+)
                from langchain.schema import SystemMessage, HumanMessage
                messages = [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=prompt)
                ]
                response = self.llm.invoke(messages)
                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)
            except:
                # Fallback to older API
                response_text = self.llm.predict(prompt)
            
            # Store in conversation memory
            if conversation_id:
                if conversation_id not in self.conversation_memory:
                    self.conversation_memory[conversation_id] = []
                self.conversation_memory[conversation_id].append({
                    "query": query,
                    "response": response_text
                })
            
            return response_text, sources if sources else None
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}", None

