"""Plain text document parser implementation."""
from pathlib import Path
from typing import List, Dict
import re

from .base_parser import BaseParser, DocumentSection, DocumentType


class TextParser(BaseParser):
    """Parser for plain text documents."""
    
    @property
    def supported_types(self) -> List[DocumentType]:
        return [DocumentType.TXT]
    
    async def parse(self, file_path: str | Path) -> List[DocumentSection]:
        """Parse plain text document into sections."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Simple section splitting - split on double newlines
            sections = []
            parts = re.split(r'\n\s*\n', content.strip())
            
            for i, part in enumerate(parts, 1):
                if not part.strip():
                    continue
                    
                # Extract first line as title if it looks like a heading
                lines = part.split('\n')
                first_line = lines[0].strip()
                
                # Check if first line looks like a heading
                if (len(lines) > 1 and 
                    (first_line.endswith(':') or 
                     len(first_line) < 80 and 
                     not first_line.endswith('.') and 
                     not first_line.endswith(','))):
                    title = first_line
                    content = '\n'.join(lines[1:]).strip()
                else:
                    title = f"Section {i}"
                    content = part.strip()
                
                sections.append(DocumentSection(
                    title=title,
                    content=content,
                    section_type="section"
                ))
            
            return sections or [DocumentSection(
                title="Document",
                content=content.strip() or "Empty document",
                section_type="document"
            )]
            
        except Exception as e:
            raise ValueError(f"Failed to parse text file: {str(e)}")
    
    async def extract_metadata(self, file_path: str | Path) -> Dict:
        """Extract basic metadata from text file."""
        try:
            # Get basic file stats
            path = Path(file_path)
            stat = path.stat()
            
            # Read first few lines to try to detect title
            title = path.stem
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    first_line = f.readline().strip()
                    if first_line and len(first_line) < 100:  # Reasonable title length
                        title = first_line
            except:
                pass
            
            return {
                "format": "TXT",
                "title": title,
                "size_bytes": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "encoding": "utf-8"
            }
        except Exception as e:
            raise ValueError(f"Failed to extract text file metadata: {str(e)}")
