"""PDF document parser implementation with advanced structure detection."""
import io
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import fitz  # PyMuPDF
from .base_parser import BaseParser, DocumentSection, DocumentType

# Regular expressions for common section headers
SECTION_PATTERNS = [
    (r'^\s*\d+\.\s+[A-Z][a-zA-Z\s]+$', 'section'),
    (r'^\s*[A-Z][A-Z\s]+$', 'heading'),
    (r'^\s*\d+\.\d+\s+[A-Z]', 'subsection'),
    (r'^\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:', 'label')
]

@dataclass
class TextBlock:
    text: str
    bbox: 'fitz.Rect'
    font_size: float
    is_bold: bool = False
    is_italic: bool = False


class PDFParser(BaseParser):
    """Parser for PDF documents using PyMuPDF with advanced structure detection."""
    
    @property
    def supported_types(self) -> List[DocumentType]:
        return [DocumentType.PDF]
    
    def _is_section_header(self, text: str) -> Tuple[bool, str]:
        """Check if text matches a section header pattern."""
        text = text.strip()
        for pattern, section_type in SECTION_PATTERNS:
            if re.match(pattern, text, re.MULTILINE):
                return True, section_type
        return False, ""
    
    def _extract_text_blocks(self, page: 'fitz.Page') -> List[TextBlock]:
        """Extract text blocks with formatting information."""
        blocks = []
        for block in page.get_text("dict").get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        blocks.append(TextBlock(
                            text=span["text"].strip(),
                            bbox=fitz.Rect(span["bbox"]),
                            font_size=span["size"],
                            is_bold=bool(span["font"].endswith("Bold")),
                            is_italic=bool("Italic" in span["font"])
                        ))
        return blocks
    
    def _group_into_sections(self, blocks: List[TextBlock]) -> List[Dict]:
        """Group text blocks into logical sections based on formatting and position."""
        if not blocks:
            return []
            
        sections = []
        current_section = {"title": "", "content": [], "metadata": {}}
        
        # Find the most common font size for body text
        font_sizes = [b.font_size for b in blocks]
        common_size = max(set(font_sizes), key=font_sizes.count)
        
        for block in blocks:
            is_header = (block.is_bold and block.font_size > common_size) or \
                       (block.is_bold and block.font_size == common_size and block.text.isupper())
            
            if is_header:
                # Save previous section if exists
                if current_section["title"] or current_section["content"]:
                    sections.append(current_section)
                
                # Start new section
                is_section, section_type = self._is_section_header(block.text)
                current_section = {
                    "title": block.text,
                    "content": [],
                    "type": section_type if is_section else "paragraph",
                    "metadata": {
                        "font_size": block.font_size,
                        "is_bold": block.is_bold,
                        "is_italic": block.is_italic
                    }
                }
            else:
                current_section["content"].append(block.text)
        
        # Add the last section
        if current_section["title"] or current_section["content"]:
            sections.append(current_section)
            
        return sections
    
    async def parse(self, file_path: str | Path) -> List[DocumentSection]:
        """Parse PDF document into structured sections with hierarchy."""
        sections = []
        
        try:
            doc = fitz.open(file_path)
            
            for page_num, page in enumerate(doc, 1):
                # Extract text blocks with formatting
                blocks = self._extract_text_blocks(page)
                
                # Group into logical sections
                page_sections = self._group_into_sections(blocks)
                
                # Convert to DocumentSection objects
                for i, section in enumerate(page_sections, 1):
                    content = '\n'.join(section["content"]).strip()
                    if not content and not section["title"]:
                        continue
                        
                    section_obj = DocumentSection(
                        title=section["title"] or f"Section {i}",
                        content=content,
                        page_number=page_num,
                        section_type=section["type"],
                        metadata={
                            **section["metadata"],
                            "page_width": page.rect.width,
                            "page_height": page.rect.height
                        }
                    )
                    sections.append(section_obj)
                
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {str(e)}")
            
        return sections
    
    async def extract_metadata(self, file_path: str | Path) -> Dict:
        """Extract metadata from PDF."""
        try:
            doc = fitz.open(file_path)
            metadata = {
                "format": "PDF",
                "page_count": len(doc),
                "author": doc.metadata.get("author"),
                "title": doc.metadata.get("title") or Path(file_path).name,
                "created": doc.metadata.get("creationDate"),
                "modified": doc.metadata.get("modDate"),
            }
            return {k: v for k, v in metadata.items() if v is not None}
        except Exception as e:
            raise ValueError(f"Failed to extract PDF metadata: {str(e)}")
        finally:
            if 'doc' in locals():
                doc.close()
