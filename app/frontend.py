"""
Streamlit Frontend for RAG Chatbot
"""

import streamlit as st
import requests
import os
from datetime import datetime
import uuid

# Page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

# API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Initialize session state
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = None

# Sidebar
with st.sidebar:
    st.title("🤖 RAG Chatbot")
    
    # API connection check
    if st.button("🔌 Check API Connection"):
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ Connected to API")
                st.session_state.api_connected = True
            else:
                st.error("❌ API not responding")
                st.session_state.api_connected = False
        except Exception as e:
            st.error(f"❌ Cannot connect: {str(e)}")
            st.session_state.api_connected = False
    
    # Status
    if st.session_state.api_connected:
        try:
            status = requests.get(f"{API_URL}/api/knowledge-base/status", timeout=5).json()
            st.info(f"📚 Knowledge Base: {'Ready' if status['ready'] else 'Not Ready'}")
            st.info(f"📄 Documents: {status.get('documents_count', 0)}")
        except:
            pass
    
    st.markdown("---")
    
    # Document upload
    st.subheader("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=['pdf', 'docx', 'txt'],
        help="Upload a document to add to the knowledge base"
    )
    
    if uploaded_file is not None:
        if st.button("📤 Upload & Process"):
            with st.spinner("Processing document..."):
                try:
                    # Reset file pointer to beginning
                    uploaded_file.seek(0)
                    files = {'file': (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
                    response = requests.post(
                        f"{API_URL}/api/upload-document",
                        files=files,
                        timeout=300
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.success(result['message'])
                        # Show chunk count if available
                        if 'chunks' in result:
                            st.info(f"Created {result['chunks']} text chunks")
                    else:
                        # Try to get error message from response
                        try:
                            error_detail = response.json().get('detail', 'Unknown error')
                            st.error(f"Upload failed: {error_detail}")
                        except:
                            st.error(f"Upload failed with status {response.status_code}: {response.text}")
                except requests.exceptions.Timeout:
                    st.error("Upload timed out. Please try again.")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Make sure backend is running on http://localhost:8000")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    st.markdown("---")
    st.info("💡 Tip: Upload documents first, then ask questions about them!")

# Main chat interface
st.title("🤖 RAG Chatbot")
st.markdown("Ask questions based on your uploaded documents!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.text(source)

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/api/chat",
                    json={
                        "message": prompt,
                        "conversation_id": st.session_state.conversation_id,
                        "use_rag": True
                    },
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    assistant_response = result["response"]
                    sources = result.get("sources")
                    
                    st.markdown(assistant_response)
                    
                    if sources:
                        with st.expander("📚 Sources"):
                            for source in sources:
                                st.text(source)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_response,
                        "sources": sources
                    })
                else:
                    st.error("Error getting response")
            except requests.exceptions.Timeout:
                st.error("Request timed out. Please try again.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure backend is running.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

