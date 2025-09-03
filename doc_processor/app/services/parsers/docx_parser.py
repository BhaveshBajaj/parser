"""DOCX document parser implementation."""
from pathlib import Path
from typing import List, Dict
import io

from docx import Document as DocxDocument
from docx.document import Document as DocumentType
from docx.table import Table
from docx.text.paragraph import Paragraph

from .base_parser import BaseParser, DocumentSection, DocumentType


class DOCXParser(BaseParser):
    """Parser for DOCX documents using python-docx."""
    
    @property
    def supported_types(self) -> List[DocumentType]:
        return [DocumentType.DOCX]
    
    async def parse(self, file_path: str | Path) -> List[DocumentSection]:
        """Parse DOCX document into sections."""
        sections = []
        current_section = None
        
        try:
            doc = DocxDocument(file_path)
            
            for element in self._iter_docx_elements(doc):
                if isinstance(element, Paragraph):
                    style = element.style.name.lower()
                    text = element.text.strip()
                    
                    if not text:
                        continue
                        
                    # Check for headings (style names typically contain 'heading')
                    if 'heading' in style:
                        # Save previous section if exists
                        if current_section and current_section.content.strip():
                            sections.append(current_section)
                        
                        # Start new section
                        current_section = DocumentSection(
                            title=text,
                            content="",
                            section_type=f"heading_{style}"
                        )
                    elif current_section:
                        # Add to current section
                        if current_section.content:
                            current_section.content += "\n\n"
                        current_section.content += text
                    else:
                        # No current section, start one with the first paragraph as title
                        current_section = DocumentSection(
                            title=text[:100] + "..." if len(text) > 100 else text,
                            content=text,
                            section_type="document"
                        )
                    
                elif isinstance(element, Table):
                    # Handle tables - convert to markdown
                    table_md = self._table_to_markdown(element)
                    if current_section:
                        if current_section.content:
                            current_section.content += "\n\n"
                        current_section.content += table_md
            
            # Add the last section if it exists
            if current_section and current_section.content.strip():
                sections.append(current_section)
                
        except Exception as e:
            raise ValueError(f"Failed to parse DOCX: {str(e)}")
            
        return sections or [DocumentSection(
            title="Document",
            content="No structured content found",
            section_type="document"
        )]
    
    def _iter_docx_elements(self, doc: DocumentType):
        """Iterate through all elements in the document."""
        for element in doc.element.body:
            if element.tag.endswith('p'):
                yield Paragraph(element, doc)
            elif element.tag.endswith('tbl'):
                yield Table(element, doc)
    
    def _table_to_markdown(self, table: Table) -> str:
        """Convert a docx table to markdown format."""
        rows = []
        for i, row in enumerate(table.rows):
            cells = [cell.text.strip() for cell in row.cells]
            rows.append('| ' + ' | '.join(cells) + ' |')
            
            # Add header separator after first row
            if i == 0:
                rows.append('|' + '|'.join(['---'] * len(row.cells)) + '|')
                
        return '\n'.join(rows)
    
    async def extract_metadata(self, file_path: str | Path) -> Dict:
        """Extract metadata from DOCX."""
        try:
            doc = DocxDocument(file_path)
            core_props = doc.core_properties
            
            return {
                "format": "DOCX",
                "author": core_props.author,
                "title": core_props.title or Path(file_path).name,
                "created": core_props.created,
                "modified": core_props.modified,
                "last_modified_by": core_props.last_modified_by,
                "revision": core_props.revision,
                "category": core_props.category,
                "comments": core_props.comments,
                "identifier": core_props.identifier,
                "keywords": core_props.keywords,
                "language": core_props.language,
                "subject": core_props.subject,
                "version": core_props.version,
            }
        except Exception as e:
            raise ValueError(f"Failed to extract DOCX metadata: {str(e)}")
