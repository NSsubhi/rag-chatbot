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
                embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
                if api_key:
                    self.embeddings = OpenAIEmbeddings(
                        model=embedding_model,
                        openai_api_key=api_key
                    )
                    logger.info(f"Using OpenAI embeddings: {embedding_model}")
                    return
            except ImportError:
                pass
            
            # Fallback to HuggingFace
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Using HuggingFace embeddings")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
    
    def _initialize_llm(self):
        """Initialize LLM (OpenAI, HuggingFace, or Ollama)"""
        try:
            # Try OpenAI first (if API key is provided)
            try:
                from langchain_openai import ChatOpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
                if api_key and api_key.strip() and not api_key.startswith("your_"):
                    # Try newer API first (model parameter)
                    try:
                        self.llm = ChatOpenAI(
                            model=model_name,
                            api_key=api_key,
                            temperature=0.7,
                            timeout=90,
                            max_retries=2
                        )
                    except TypeError:
                        # Fallback to older API (model_name and openai_api_key)
                        self.llm = ChatOpenAI(
                            model_name=model_name,
                            openai_api_key=api_key,
                            temperature=0.7,
                            timeout=90,
                            max_retries=2
                        )
                    logger.info(f"Using OpenAI ChatOpenAI: {model_name}")
                    return
                else:
                    logger.info("OpenAI API key not provided or invalid, using free LLM")
            except ImportError:
                logger.info("langchain-openai not available, using free LLM")
            except Exception as e:
                logger.warning(f"OpenAI initialization failed: {e}, using free LLM")
            
            # Try HuggingFace Inference API (free tier available)
            try:
                from langchain_community.llms import HuggingFacePipeline, HuggingFaceEndpoint
                
                # Try HuggingFace Inference API first (free tier)
                hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
                if hf_token:
                    try:
                        # Use a free model from HuggingFace
                        model_id = os.getenv("HUGGINGFACE_MODEL", "google/flan-t5-base")
                        self.llm = HuggingFaceEndpoint(
                            repo_id=model_id,
                            huggingfacehub_api_token=hf_token,
                            temperature=0.7,
                            max_length=512
                        )
                        logger.info(f"Using HuggingFace Inference API: {model_id}")
                        return
                    except Exception as e:
                        logger.warning(f"HuggingFace Inference API failed: {e}")
                
                # Fallback to local HuggingFace pipeline (completely free, no API needed)
                try:
                    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
                    import torch
                    
                    # Use a smaller, faster model for local inference
                    model_name = os.getenv("LOCAL_MODEL", "microsoft/DialoGPT-small")
                    logger.info(f"Loading local HuggingFace model: {model_name} (this may take a moment...)")
                    
                    # Load model (this will download on first run)
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(model_name)
                    
                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_length=512,
                        temperature=0.7,
                        device_map="auto" if torch.cuda.is_available() else None
                    )
                    
                    self.llm = HuggingFacePipeline(pipeline=pipe)
                    logger.info(f"Using local HuggingFace model: {model_name}")
                    return
                except ImportError:
                    logger.info("transformers not installed, trying Ollama...")
                except Exception as e:
                    logger.warning(f"Local HuggingFace model failed: {e}, trying Ollama...")
            except ImportError:
                logger.info("HuggingFace not available, trying Ollama...")
            
            # Fallback to Ollama (requires local installation)
            try:
                try:
                    from langchain_ollama import OllamaLLM
                    ollama_model = os.getenv("OLLAMA_MODEL", "llama2")
                    self.llm = OllamaLLM(model=ollama_model)
                    logger.info(f"Using Ollama (local) via langchain-ollama: {ollama_model}")
                    return
                except ImportError:
                    from langchain_community.llms import Ollama
                    ollama_model = os.getenv("OLLAMA_MODEL", "llama2")
                    self.llm = Ollama(model=ollama_model)
                    logger.info(f"Using Ollama (local) via langchain-community: {ollama_model}")
                    return
            except ImportError:
                logger.warning("Ollama not available. Install: pip install langchain-community ollama")
            except Exception as e:
                logger.warning(f"Ollama initialization failed: {e}")
            
            # If all else fails, use a simple text completion fallback
            logger.warning("No LLM could be initialized. Please install one of: transformers, ollama, or provide OpenAI API key")
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
    
    def _initialize_vector_store(self):
        """Initialize Chroma vector store"""
        try:
            import chromadb
            # Try newest import path first (langchain-chroma)
            try:
                from langchain_chroma import Chroma
            except ImportError:
                # Try newer import path (langchain-community)
                try:
                    from langchain_community.vectorstores import Chroma
                except ImportError:
                    # Fallback to older import path
                    from langchain.vectorstores import Chroma
            
            # Initialize Chroma
            persist_directory = os.getenv("CHROMA_DB_PATH", "./chroma_db")
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
                # Check if LLM is a ChatOpenAI model (expects messages)
                llm_type = type(self.llm).__name__
                if 'ChatOpenAI' in llm_type or 'OpenAI' in llm_type:
                    # Try newer LangChain API (0.1.x+)
                    try:
                        from langchain_core.messages import SystemMessage, HumanMessage
                    except ImportError:
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
                else:
                    # For HuggingFace, Ollama, and other models - use invoke with string
                    response = self.llm.invoke(prompt)
                    if hasattr(response, 'content'):
                        response_text = response.content
                    else:
                        response_text = str(response)
            except Exception as e:
                error_str = str(e)
                # Check for quota/rate limit errors
                if '429' in error_str or 'quota' in error_str.lower() or 'insufficient_quota' in error_str.lower():
                    logger.error(f"OpenAI API quota exceeded: {e}")
                    return (
                        "I'm sorry, but the OpenAI API quota has been exceeded. Please check your OpenAI account billing and quota settings. "
                        "You can visit https://platform.openai.com/account/billing to check your usage and billing information.",
                        sources if sources else None
                    )
                
                logger.warning(f"Newer API failed, trying fallback: {e}")
                # Fallback to older API
                try:
                    # For ChatOpenAI, we need to use invoke with messages
                    # Try with messages format
                    try:
                        from langchain_core.messages import HumanMessage
                        response = self.llm.invoke([HumanMessage(content=prompt)])
                        response_text = response.content if hasattr(response, 'content') else str(response)
                    except:
                        # Try using predict method if available
                        if hasattr(self.llm, 'predict'):
                            response_text = self.llm.predict(prompt)
                        else:
                            # Last resort: try invoke with string
                            response = self.llm.invoke(prompt)
                            response_text = response.content if hasattr(response, 'content') else str(response)
                except Exception as fallback_error:
                    logger.error(f"All fallback methods failed: {fallback_error}")
                    return (
                        f"Error generating response: {str(fallback_error)}. Please check your OpenAI API key and quota.",
                        sources if sources else None
                    )
            
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

