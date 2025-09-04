"""Document processing service that coordinates parsing and entity extraction."""
import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4

from config.settings import settings
from models.document import Document, DocumentStatus, DocumentUpdate
from services.parsers import DocumentSection
from services.langchain_parsers import LangChainParserFactory
from services.embedding_service import EmbeddingService
from services.agents.autogen_agents import AutoGenWorkflowOrchestrator

logger = logging.getLogger(__name__)


class DocumentProcessor:
    
    def __init__(self):
        self.autogen_orchestrator = AutoGenWorkflowOrchestrator()
        self.embedding_service = EmbeddingService()
        self.documents: Dict[UUID, Document] = {}
        self._ensure_upload_dir()
    
    def _ensure_upload_dir(self) -> None:
        os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    
    async def process_document(self, file_path: str | Path, filename: str, content_type: str) -> Document:

        doc = Document(
            filename=filename,
            content_type=content_type,
            size=os.path.getsize(file_path),
            status=DocumentStatus.PROCESSING
        )
        
        try:
            # Get the appropriate LangChain parser
            parser = LangChainParserFactory.get_parser_for_type(content_type)
            logger.info(f"Using LangChain parser for {content_type}")
            
            # Extract metadata
            metadata = await parser.extract_metadata(file_path)
            doc.metadata.update(metadata)
            
            sections = await parser.parse(file_path)
            full_text = "\n\n".join(section.content for section in sections)
            
            # Add sections to document before calling workflow so agents can access content
            doc.extra_data = {
                "sections": [{
                    "title": section.title,
                    "content": section.content,
                    "section_type": section.section_type,
                    "page_number": section.page_number,
                    "metadata": section.metadata
                } for section in sections]
            }
            
            # Run full workflow to get both summarization and entity extraction
            workflow_result = await self.autogen_orchestrator.execute_document_workflow(
                document=doc,
                workflow_type="full"
            )
            
            entities = self._extract_entities_from_autogen_result(workflow_result)
            summary = self._extract_summary_from_autogen_result(workflow_result)
            
            embedded_sections = await self._generate_section_embeddings(sections)
            
            # Use the actual summary from the workflow, or fallback to generic message
            doc.summary = summary if summary else f"Document processed with {len(sections)} sections and {len(entities)} entities"
            doc.metadata.update({
                "sections": len(sections),
                "entities": len(entities),
                "entity_types": self._count_entity_types(entities),
                "embeddings_generated": len(embedded_sections),
                "langchain_processed": True
            })
            
            # Store the full results (merge with existing extra_data)
            doc.extra_data.update({
                "all_entities": entities,  # entities are already dictionaries
                "embedded_sections": embedded_sections
            })
            
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
    
    
    def _extract_entities_from_autogen_result(self, workflow_result: Dict[str, Any]) -> List:
    
        try:
            # Get entity agent results
            agent_results = workflow_result.get("agent_results", {})
            entity_agent_result = agent_results.get("entity_agent", {})
            
            if not entity_agent_result:
                logger.warning("No entity agent results found in AutoGen workflow")
                logger.debug(f"Available agent results: {list(agent_results.keys())}")
                return []
            
            result_data = entity_agent_result.get("result_data", {})
            raw_response = entity_agent_result.get("raw_response", "")
            
            logger.debug(f"Entity agent result_data: {result_data}")
            logger.debug(f"Entity agent raw_response: {raw_response[:200]}...")
            
            entities_data = result_data.get("entities", [])
            
            if not entities_data:
                logger.warning("No entities found in entity agent result_data")
                logger.debug(f"Result data keys: {list(result_data.keys())}")
                return []
            
            entities = []
            for entity_data in entities_data:
                if isinstance(entity_data, dict):
                    entity = {
                        "text": entity_data.get("text", ""),
                        "type": entity_data.get("type", "MISC"),
                        "start": entity_data.get("start_pos", 0),
                        "end": entity_data.get("end_pos", 0),
                        "confidence": entity_data.get("confidence", 0.5),
                        "metadata": entity_data.get("metadata", {})
                    }
                    entities.append(entity)
                else:
                    logger.warning(f"Invalid entity data format: {entity_data}")
            
            logger.info(f"Extracted {len(entities)} entities from AutoGen workflow")
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities from AutoGen result: {str(e)}", exc_info=True)
            return []
    
    def _extract_summary_from_autogen_result(self, workflow_result: Dict[str, Any]) -> str:
        """Extract summary from AutoGen workflow results."""
        try:
            # Get summarization agent results
            agent_results = workflow_result.get("agent_results", {})
            summarization_agent_result = agent_results.get("summarization_agent", {})
            
            if not summarization_agent_result:
                logger.warning("No summarization agent results found in AutoGen workflow")
                return ""
            
            result_data = summarization_agent_result.get("result_data", {})
            summary = result_data.get("summary", "")
            
            if summary:
                logger.info(f"Extracted summary from AutoGen workflow: {len(summary)} characters")
                return summary
            else:
                logger.warning("No summary found in summarization agent result_data")
                return ""
                
        except Exception as e:
            logger.error(f"Error extracting summary from AutoGen result: {str(e)}", exc_info=True)
            return ""
    
    async def _generate_section_embeddings(self, sections: List[DocumentSection]) -> List[Dict[str, Any]]:
       
        try:
            # Convert sections to the format expected by embedding service
            section_dicts = []
            for section in sections:
                section_dicts.append({
                    'title': section.title,
                    'content': section.content
                })
            
            # Generate embeddings using the embedding service
            embedded_sections = await self.embedding_service.embed_document_sections(section_dicts)
            
            logger.info(f"Generated embeddings for {len(embedded_sections)} sections")
            return embedded_sections
            
        except Exception as e:
            logger.error(f"Failed to generate section embeddings: {str(e)}")
            return []
    
    
    
    
    def _count_entity_types(self, entities: List[Dict]) -> Dict[str, int]:
        """Count entities by type."""
        counts = {}
        for entity in entities:
            entity_type = entity.get("type", "MISC")
            counts[entity_type] = counts.get(entity_type, 0) + 1
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
