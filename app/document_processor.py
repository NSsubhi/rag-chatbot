"""
Document Processor - Handles document parsing and chunking
"""

import os
import logging
from typing import List
from io import BytesIO

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_file(self, filename: str, content: bytes) -> List[str]:
        """Process uploaded file and return text chunks"""
        file_ext = os.path.splitext(filename)[1].lower()
        
        try:
            if file_ext == '.pdf':
                return self._process_pdf(content)
            elif file_ext in ['.docx', '.doc']:
                return self._process_docx(content)
            elif file_ext == '.txt':
                return self._process_txt(content)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            raise
    
    def _process_pdf(self, content: bytes) -> List[str]:
        """Process PDF file"""
        try:
            from pypdf import PdfReader
            pdf_reader = PdfReader(BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return self._chunk_text(text)
        except ImportError:
            raise ImportError("pypdf not installed. Install: pip install pypdf")
    
    def _process_docx(self, content: bytes) -> List[str]:
        """Process DOCX file"""
        try:
            from docx import Document
            doc = Document(BytesIO(content))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return self._chunk_text(text)
        except ImportError:
            raise ImportError("python-docx not installed. Install: pip install python-docx")
    
    def _process_txt(self, content: bytes) -> List[str]:
        """Process TXT file"""
        try:
            text = content.decode('utf-8')
            return self._chunk_text(text)
        except UnicodeDecodeError:
            text = content.decode('latin-1')
            return self._chunk_text(text)
    
    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text into smaller pieces"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        chunks = text_splitter.split_text(text)
        return chunks

