"""Base parser interface for document processing."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    HTML = "text/html"
    TXT = "text/plain"


@dataclass
class DocumentSection:
    """Represents a section in a document."""
    title: str
    content: str
    page_number: Optional[int] = None
    section_type: str = "section"
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseParser(ABC):
    """Abstract base class for document parsers."""
    
    @property
    @abstractmethod
    def supported_types(self) -> List[DocumentType]:
        """List of document types supported by this parser."""
        pass
    
    @abstractmethod
    async def parse(self, file_path: str | Path) -> List[DocumentSection]:
        """
        Parse the document and return a list of sections.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of document sections with their content
        """
        pass
    
    @abstractmethod
    async def extract_metadata(self, file_path: str | Path) -> Dict:
        """Extract metadata from the document."""
        pass
    
    def get_supported_extensions(self) -> List[str]:
        """Get file extensions supported by this parser."""
        return [f".{t.value.split('/')[-1]}" for t in self.supported_types]
    
    @classmethod
    def get_parser_for_type(cls, content_type: str) -> 'BaseParser':
        """Get the appropriate parser for the given content type using LangChain."""
        from services.langchain_parsers import LangChainParserFactory
        
        return LangChainParserFactory.get_parser_for_type(content_type)
