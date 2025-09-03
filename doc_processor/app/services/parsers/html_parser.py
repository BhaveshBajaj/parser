"""HTML document parser implementation."""
from pathlib import Path
from typing import List, Dict
from bs4 import BeautifulSoup, Tag, NavigableString
import re

from .base_parser import BaseParser, DocumentSection, DocumentType


class HTMLParser(BaseParser):
    """Parser for HTML documents using BeautifulSoup."""
    
    @property
    def supported_types(self) -> List[DocumentType]:
        return [DocumentType.HTML]
    
    async def parse(self, file_path: str | Path) -> List[DocumentSection]:
        """Parse HTML document into sections."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Process headings and content
            sections = []
            current_section = None
            
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'table']):
                if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    # Save previous section
                    if current_section and current_section.content.strip():
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = DocumentSection(
                        title=element.get_text(strip=True),
                        content="",
                        section_type=f"heading_{element.name}"
                    )
                    
                elif current_section:
                    # Add content to current section
                    text = self._get_element_text(element)
                    if text.strip():
                        if current_section.content:
                            current_section.content += "\n\n"
                        current_section.content += text
            
            # Add the last section if it exists
            if current_section and current_section.content.strip():
                sections.append(current_section)
                
            return sections or [DocumentSection(
                title=soup.title.string if soup.title else "HTML Document",
                content=self._get_element_text(soup.body) if soup.body else "No content found",
                section_type="document"
            )]
            
        except Exception as e:
            raise ValueError(f"Failed to parse HTML: {str(e)}")
    
    def _get_element_text(self, element) -> str:
        """Extract text from an HTML element, handling various content types."""
        if element.name == 'table':
            return self._table_to_markdown(element)
        return element.get_text('\n', strip=True)
    
    def _table_to_markdown(self, table: Tag) -> str:
        """Convert an HTML table to markdown format."""
        rows = []
        headers = []
        
        # Process header row if it exists
        thead = table.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                rows.append('| ' + ' | '.join(headers) + ' |')
                rows.append('|' + '|'.join(['---'] * len(headers)) + '|')
        
        # Process body rows
        tbody = table.find('tbody') or table
        for row in tbody.find_all('tr'):
            # Skip header row if we already processed it
            if row == thead.find('tr') if thead else False:
                continue
                
            cells = [self._get_cell_text(cell) for cell in row.find_all(['td', 'th'])]
            if not headers and not rows and cells:
                # This is the first row and we didn't find a header, use it as header
                headers = cells
                rows.append('| ' + ' | '.join(headers) + ' |')
                rows.append('|' + '|'.join(['---'] * len(headers)) + '|')
            else:
                rows.append('| ' + ' | '.join(cells) + ' |')
        
        return '\n'.join(rows)
    
    def _get_cell_text(self, cell: Tag) -> str:
        """Extract text from a table cell, handling nested elements."""
        return ' '.join(cell.stripped_strings)
    
    async def extract_metadata(self, file_path: str | Path) -> Dict:
        """Extract metadata from HTML."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            meta = {}
            
            # Get title
            title = soup.title.string if soup.title else Path(file_path).name
            
            # Get meta tags
            for tag in soup.find_all('meta'):
                name = tag.get('name') or tag.get('property') or tag.get('http-equiv')
                if name:
                    meta[name.lower()] = tag.get('content', '')
            
            return {
                "format": "HTML",
                "title": title,
                "meta": meta,
                "language": soup.get('lang') or soup.find('html', {}).get('lang'),
                "charset": soup.meta.get('charset') if soup.meta else None,
            }
        except Exception as e:
            raise ValueError(f"Failed to extract HTML metadata: {str(e)}")
