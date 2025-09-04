"""LangChain-based document parsers for enhanced document processing."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredHTMLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

from .parsers.base_parser import BaseParser, DocumentSection, DocumentType

logger = logging.getLogger(__name__)


@dataclass
class LangChainDocumentSection(DocumentSection):
    """Extended document section with LangChain metadata."""
    source: Optional[str] = None
    langchain_metadata: Optional[Dict[str, Any]] = None


class LangChainPDFParser(BaseParser):
    """LangChain-based PDF parser with enhanced document processing."""
    
    @property
    def supported_types(self) -> List[DocumentType]:
        return [DocumentType.PDF]
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    async def parse(self, file_path: str | Path) -> List[DocumentSection]:
        """Parse PDF document using LangChain PyPDFLoader."""
        try:
            # Load document using LangChain
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)
            
            # Convert to DocumentSection objects
            sections = []
            for i, doc in enumerate(split_docs):
                # Extract page number from metadata
                page_number = doc.metadata.get('page', i + 1)
                
                # Create section title from first few words or page info
                content_preview = doc.page_content[:100].strip()
                title = f"Page {page_number}" if not content_preview else content_preview.split('\n')[0][:50]
                
                section = DocumentSection(
                    title=title,
                    content=doc.page_content,
                    page_number=page_number,
                    section_type="page_chunk",
                    metadata={
                        "source": doc.metadata.get('source', str(file_path)),
                        "chunk_id": i,
                        "total_chunks": len(split_docs),
                        "langchain_metadata": doc.metadata
                    }
                )
                sections.append(section)
            
            logger.info(f"Parsed PDF into {len(sections)} sections using LangChain")
            return sections
            
        except Exception as e:
            logger.error(f"LangChain PDF parsing failed: {str(e)}")
            raise ValueError(f"Failed to parse PDF with LangChain: {str(e)}")
    
    async def extract_metadata(self, file_path: str | Path) -> Dict:
        """Extract metadata from PDF using LangChain."""
        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            # Get metadata from first document
            if documents:
                metadata = documents[0].metadata
                return {
                    "format": "PDF",
                    "source": metadata.get('source', str(file_path)),
                    "page_count": len(documents),
                    "total_chunks": len(self.text_splitter.split_documents(documents)),
                    "langchain_processed": True
                }
            else:
                return {"format": "PDF", "langchain_processed": True}
                
        except Exception as e:
            logger.error(f"Failed to extract PDF metadata with LangChain: {str(e)}")
            return {"format": "PDF", "error": str(e)}


class LangChainDOCXParser(BaseParser):
    """LangChain-based DOCX parser with enhanced document processing."""
    
    @property
    def supported_types(self) -> List[DocumentType]:
        return [DocumentType.DOCX]
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    async def parse(self, file_path: str | Path) -> List[DocumentSection]:
        """Parse DOCX document using LangChain Docx2txtLoader."""
        try:
            # Load document using LangChain
            loader = Docx2txtLoader(str(file_path))
            documents = loader.load()
            
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)
            
            # Convert to DocumentSection objects
            sections = []
            for i, doc in enumerate(split_docs):
                # Create section title from first few words
                content_preview = doc.page_content[:100].strip()
                title = content_preview.split('\n')[0][:50] if content_preview else f"Section {i + 1}"
                
                section = DocumentSection(
                    title=title,
                    content=doc.page_content,
                    section_type="docx_chunk",
                    metadata={
                        "source": doc.metadata.get('source', str(file_path)),
                        "chunk_id": i,
                        "total_chunks": len(split_docs),
                        "langchain_metadata": doc.metadata
                    }
                )
                sections.append(section)
            
            logger.info(f"Parsed DOCX into {len(sections)} sections using LangChain")
            return sections
            
        except Exception as e:
            logger.error(f"LangChain DOCX parsing failed: {str(e)}")
            raise ValueError(f"Failed to parse DOCX with LangChain: {str(e)}")
    
    async def extract_metadata(self, file_path: str | Path) -> Dict:
        """Extract metadata from DOCX using LangChain."""
        try:
            loader = Docx2txtLoader(str(file_path))
            documents = loader.load()
            
            # Get metadata from first document
            if documents:
                metadata = documents[0].metadata
                return {
                    "format": "DOCX",
                    "source": metadata.get('source', str(file_path)),
                    "total_chunks": len(self.text_splitter.split_documents(documents)),
                    "langchain_processed": True
                }
            else:
                return {"format": "DOCX", "langchain_processed": True}
                
        except Exception as e:
            logger.error(f"Failed to extract DOCX metadata with LangChain: {str(e)}")
            return {"format": "DOCX", "error": str(e)}


class LangChainTextParser(BaseParser):
    """LangChain-based text parser with enhanced document processing."""
    
    @property
    def supported_types(self) -> List[DocumentType]:
        return [DocumentType.TXT]
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    async def parse(self, file_path: str | Path) -> List[DocumentSection]:
        """Parse text document using LangChain TextLoader."""
        try:
            # Load document using LangChain
            loader = TextLoader(str(file_path), encoding='utf-8')
            documents = loader.load()
            
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)
            
            # Convert to DocumentSection objects
            sections = []
            for i, doc in enumerate(split_docs):
                # Create section title from first few words
                content_preview = doc.page_content[:100].strip()
                title = content_preview.split('\n')[0][:50] if content_preview else f"Section {i + 1}"
                
                section = DocumentSection(
                    title=title,
                    content=doc.page_content,
                    section_type="text_chunk",
                    metadata={
                        "source": doc.metadata.get('source', str(file_path)),
                        "chunk_id": i,
                        "total_chunks": len(split_docs),
                        "langchain_metadata": doc.metadata
                    }
                )
                sections.append(section)
            
            logger.info(f"Parsed text file into {len(sections)} sections using LangChain")
            return sections
            
        except Exception as e:
            logger.error(f"LangChain text parsing failed: {str(e)}")
            raise ValueError(f"Failed to parse text file with LangChain: {str(e)}")
    
    async def extract_metadata(self, file_path: str | Path) -> Dict:
        """Extract metadata from text file using LangChain."""
        try:
            loader = TextLoader(str(file_path), encoding='utf-8')
            documents = loader.load()
            
            # Get metadata from first document
            if documents:
                metadata = documents[0].metadata
                return {
                    "format": "TXT",
                    "source": metadata.get('source', str(file_path)),
                    "total_chunks": len(self.text_splitter.split_documents(documents)),
                    "langchain_processed": True
                }
            else:
                return {"format": "TXT", "langchain_processed": True}
                
        except Exception as e:
            logger.error(f"Failed to extract text metadata with LangChain: {str(e)}")
            return {"format": "TXT", "error": str(e)}


class LangChainHTMLParser(BaseParser):
    """LangChain-based HTML parser with enhanced document processing."""
    
    @property
    def supported_types(self) -> List[DocumentType]:
        return [DocumentType.HTML]
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    async def parse(self, file_path: str | Path) -> List[DocumentSection]:
        """Parse HTML document using LangChain UnstructuredHTMLLoader."""
        try:
            # Load document using LangChain
            loader = UnstructuredHTMLLoader(str(file_path))
            documents = loader.load()
            
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)
            
            # Convert to DocumentSection objects
            sections = []
            for i, doc in enumerate(split_docs):
                # Create section title from first few words
                content_preview = doc.page_content[:100].strip()
                title = content_preview.split('\n')[0][:50] if content_preview else f"Section {i + 1}"
                
                section = DocumentSection(
                    title=title,
                    content=doc.page_content,
                    section_type="html_chunk",
                    metadata={
                        "source": doc.metadata.get('source', str(file_path)),
                        "chunk_id": i,
                        "total_chunks": len(split_docs),
                        "langchain_metadata": doc.metadata
                    }
                )
                sections.append(section)
            
            logger.info(f"Parsed HTML into {len(sections)} sections using LangChain")
            return sections
            
        except Exception as e:
            logger.error(f"LangChain HTML parsing failed: {str(e)}")
            raise ValueError(f"Failed to parse HTML with LangChain: {str(e)}")
    
    async def extract_metadata(self, file_path: str | Path) -> Dict:
        """Extract metadata from HTML using LangChain."""
        try:
            loader = UnstructuredHTMLLoader(str(file_path))
            documents = loader.load()
            
            # Get metadata from first document
            if documents:
                metadata = documents[0].metadata
                return {
                    "format": "HTML",
                    "source": metadata.get('source', str(file_path)),
                    "total_chunks": len(self.text_splitter.split_documents(documents)),
                    "langchain_processed": True
                }
            else:
                return {"format": "HTML", "langchain_processed": True}
                
        except Exception as e:
            logger.error(f"Failed to extract HTML metadata with LangChain: {str(e)}")
            return {"format": "HTML", "error": str(e)}


class LangChainParserFactory:
    """Factory for creating LangChain-based parsers."""
    
    @staticmethod
    def get_parser_for_type(content_type: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> BaseParser:
        """Get the appropriate LangChain parser for the given content type."""
        content_type = content_type.lower()
        
        if 'pdf' in content_type:
            return LangChainPDFParser(chunk_size, chunk_overlap)
        elif 'wordprocessingml.document' in content_type or 'docx' in content_type:
            return LangChainDOCXParser(chunk_size, chunk_overlap)
        elif 'html' in content_type:
            return LangChainHTMLParser(chunk_size, chunk_overlap)
        elif 'text/plain' in content_type or 'txt' in content_type:
            return LangChainTextParser(chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unsupported content type for LangChain parser: {content_type}")
    
    @staticmethod
    def get_all_parsers(chunk_size: int = 1000, chunk_overlap: int = 200) -> List[BaseParser]:
        """Get all available LangChain parsers."""
        return [
            LangChainPDFParser(chunk_size, chunk_overlap),
            LangChainDOCXParser(chunk_size, chunk_overlap),
            LangChainHTMLParser(chunk_size, chunk_overlap),
            LangChainTextParser(chunk_size, chunk_overlap)
        ]
