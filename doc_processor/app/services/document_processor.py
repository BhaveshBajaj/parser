"""Document processing service that coordinates parsing and entity extraction."""
import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4

from ..core.config import settings
from ..models.document import Document, DocumentStatus, DocumentUpdate
from ..services.parsers import BaseParser, DocumentSection
from ..services.entity_extractor import EntityExtractor, Entity

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Service for processing documents through the parsing and entity extraction pipeline."""
    
    def __init__(self):
        """Initialize the document processor with required services."""
        self.entity_extractor = EntityExtractor()
        self.documents: Dict[UUID, Document] = {}
        self._ensure_upload_dir()
    
    def _ensure_upload_dir(self) -> None:
        """Ensure the upload directory exists."""
        os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    
    async def process_document(self, file_path: str | Path, filename: str, content_type: str) -> Document:
        """
        Process a document through the entire pipeline.
        
        Args:
            file_path: Path to the document file
            filename: Original filename
            content_type: MIME type of the file
            
        Returns:
            The processed Document object
        """
        # Create a new document record
        doc = Document(
            filename=filename,
            content_type=content_type,
            size=os.path.getsize(file_path),
            status=DocumentStatus.PROCESSING
        )
        
        try:
            # Get the appropriate parser
            parser = BaseParser.get_parser_for_type(content_type)
            
            # Extract metadata
            metadata = await parser.extract_metadata(file_path)
            doc.metadata.update(metadata)
            
            # Parse the document into sections
            sections = await parser.parse(file_path)
            
            # Extract text from sections for entity extraction
            full_text = "\n\n".join(section.content for section in sections)
            
            # Extract entities from the full text
            entities = await self.entity_extractor.extract_entities(full_text)
            
            # Group entities by section
            section_entities = self._assign_entities_to_sections(sections, entities)
            
            # Update document with results
            doc.summary = self._generate_summary(sections, entities)
            doc.metadata.update({
                "sections": len(sections),
                "entities": len(entities),
                "entity_types": self._count_entity_types(entities)
            })
            
            # Store the full results
            doc.extra_data = {
                "sections": [{
                    "title": section.title,
                    "content": section.content,
                    "section_type": section.section_type,
                    "page_number": section.page_number,
                    "entities": [e.to_dict() for e in section_entities[i]]
                } for i, section in enumerate(sections)],
                "all_entities": [e.to_dict() for e in entities]
            }
            
            doc.status = DocumentStatus.PROCESSED
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}", exc_info=True)
            doc.status = DocumentStatus.FAILED
            doc.error = str(e)
            raise
            
        finally:
            # Store the document
            self.documents[doc.id] = doc
            
        return doc
    
    def _assign_entities_to_sections(
        self, 
        sections: List[DocumentSection], 
        entities: List[Entity]
    ) -> List[List[Entity]]:
        """
        Assign entities to their respective sections based on text position.
        
        Args:
            sections: List of document sections
            entities: List of entities to assign
            
        Returns:
            List of entity lists, one per section
        """
        # Calculate section text positions
        section_positions = []
        current_pos = 0
        
        for section in sections:
            section_start = current_pos
            section_end = section_start + len(section.content)
            section_positions.append((section_start, section_end))
            current_pos = section_end + 2  # +2 for the "\n\n" between sections
        
        # Assign entities to sections
        section_entities = [[] for _ in sections]
        
        for entity in entities:
            # Find which sections this entity overlaps with
            for i, (start, end) in enumerate(section_positions):
                # Check if entity overlaps with this section
                if entity.start < end and entity.end > start:
                    # Adjust entity positions to be relative to the section
                    section_entity = Entity(
                        text=entity.text,
                        type=entity.type,
                        start=max(0, entity.start - start),
                        end=min(len(sections[i].content), entity.end - start),
                        confidence=entity.confidence,
                        metadata=entity.metadata.copy()
                    )
                    section_entities[i].append(section_entity)
        
        return section_entities
    
    def _generate_summary(
        self, 
        sections: List[DocumentSection], 
        entities: List[Entity]
    ) -> str:
        """
        Generate a summary of the document based on sections and entities using LLM.
        
        Args:
            sections: List of document sections
            entities: List of extracted entities
            
        Returns:
            A summary string
        """
        # Try LLM-based summary generation first
        try:
            llm_summary = self._generate_llm_summary(sections, entities)
            if llm_summary and len(llm_summary.strip()) > 50:  # Ensure we got a meaningful summary
                return llm_summary
        except Exception as e:
            logger.warning(f"LLM summary generation failed: {str(e)}, falling back to basic summary")
        
        # Fallback to basic summary if LLM fails
        return self._generate_basic_summary(sections, entities)
    
    def _generate_llm_summary(
        self, 
        sections: List[DocumentSection], 
        entities: List[Entity]
    ) -> str:
        """
        Generate a summary using Azure OpenAI LLM.
        
        Args:
            sections: List of document sections
            entities: List of extracted entities
            
        Returns:
            A summary string generated by LLM
        """
        from ..core.config import settings
        from openai import AzureOpenAI
        
        # Check if Azure OpenAI is configured
        if not settings.AZURE_OPENAI_API_KEY or not settings.AZURE_OPENAI_ENDPOINT:
            logger.info("Azure OpenAI not configured, using basic summary")
            raise Exception("Azure OpenAI not configured")
        
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
        )
        deployment = settings.AZURE_OPENAI_DEPLOYMENT
        
        # Prepare content for summarization
        content_parts = []
        
        # Add section content (limited to avoid token limits)
        for i, section in enumerate(sections[:10]):  # Limit to first 10 sections
            if section.content:
                # Truncate very long sections
                content = section.content[:1000] + "..." if len(section.content) > 1000 else section.content
                content_parts.append(f"Section {i+1} - {section.title or 'Untitled'}:\n{content}")
        
        # Add entity information
        entity_info = self._format_entities_for_summary(entities)
        
        document_content = "\n\n".join(content_parts)
        
        # Create the prompt
        system_message = """You are a professional document summarizer. Create a comprehensive, well-structured summary of the provided document content. Your summary should:

1. Capture the main topics and key points
2. Highlight important findings, conclusions, or recommendations
3. Mention significant entities (people, organizations, locations, etc.) when relevant
4. Be clear, concise, and informative
5. Be 2-4 paragraphs long

Focus on the substance and meaning rather than just listing sections or entity counts."""

        user_message = f"""Please summarize this document:

DOCUMENT CONTENT:
{document_content}

EXTRACTED ENTITIES:
{entity_info}

Provide a comprehensive summary that captures the document's main content, purpose, and key information."""

        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info(f"Generated LLM summary of length: {len(summary)}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating LLM summary: {str(e)}")
            raise
    
    def _format_entities_for_summary(self, entities: List[Entity]) -> str:
        """Format entities for inclusion in summary prompt."""
        if not entities:
            return "No entities extracted."
        
        # Group entities by type
        entity_groups = {}
        for entity in entities[:50]:  # Limit to avoid token limits
            entity_type = entity.type.value
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(entity.text)
        
        # Format for prompt
        formatted_parts = []
        for entity_type, entity_texts in entity_groups.items():
            unique_texts = list(set(entity_texts))[:10]  # Limit and deduplicate
            formatted_parts.append(f"{entity_type}: {', '.join(unique_texts)}")
        
        return "\n".join(formatted_parts)
    
    def _generate_basic_summary(
        self, 
        sections: List[DocumentSection], 
        entities: List[Entity]
    ) -> str:
        """
        Generate a basic summary as fallback when LLM is not available.
        
        Args:
            sections: List of document sections
            entities: List of extracted entities
            
        Returns:
            A basic summary string
        """
        # Count entity types
        entity_counts = self._count_entity_types(entities)
        
        # Get section titles
        section_titles = [s.title for s in sections if s.title]
        
        # Build summary
        summary_parts = []
        
        if section_titles:
            summary_parts.append(f"Document contains {len(sections)} sections: "
                               f"{', '.join(f'"{t}"' for t in section_titles[:5])}")
        else:
            summary_parts.append(f"Document contains {len(sections)} sections.")
        
        if entity_counts:
            entity_summary = ", ".join(f"{count} {etype}" for etype, count in entity_counts.items())
            summary_parts.append(f"Found {len(entities)} entities: {entity_summary}.")
        
        return " ".join(summary_parts)
    
    def _count_entity_types(self, entities: List[Entity]) -> Dict[str, int]:
        """Count entities by type."""
        counts = {}
        for entity in entities:
            counts[entity.type.value] = counts.get(entity.type.value, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    
    async def get_document(self, doc_id: UUID) -> Optional[Document]:
        """Get a document by ID."""
        return self.documents.get(doc_id)
    
    async def update_document(self, doc_id: UUID, update: DocumentUpdate) -> Optional[Document]:
        """Update a document."""
        if doc_id not in self.documents:
            return None
            
        doc = self.documents[doc_id]
        
        # Update fields
        if update.status:
            doc.status = update.status
        if update.summary:
            doc.summary = update.summary
        if update.metadata:
            doc.metadata.update(update.metadata)
        if update.extra_data:
            if doc.extra_data is None:
                doc.extra_data = {}
            doc.extra_data.update(update.extra_data)
            
        doc.updated_at = datetime.now(timezone.utc)
        return doc
