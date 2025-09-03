"""Summarization agents for different levels of document processing."""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from .base_agent import BaseAgent, AgentResult, AgentStatus, AgentError
from ...models.document import Document
from ..entity_extractor import Entity, EntityType

logger = logging.getLogger(__name__)


class SectionSummarizationAgent(BaseAgent):
    """Agent for summarizing individual document sections."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="section_summarizer",
            description="Summarizes individual sections of a document",
            config=config
        )
        self.max_section_length = self.config.get("max_section_length", 2000)
        self.summary_length = self.config.get("summary_length", 200)
    
    async def validate_input(self, document: Document, context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate that the document has sections to summarize."""
        if not document.extra_data or "sections" not in document.extra_data:
            self.logger.warning(f"Document {document.id} has no sections to summarize")
            return False
        
        sections = document.extra_data["sections"]
        if not sections or len(sections) == 0:
            self.logger.warning(f"Document {document.id} has empty sections")
            return False
        
        return True
    
    async def process(
        self, 
        document: Document, 
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResult:
        """Summarize each section of the document."""
        try:
            sections = document.extra_data["sections"]
            summarized_sections = []
            
            for i, section in enumerate(sections):
                section_summary = await self._summarize_section(section, i)
                summarized_sections.append({
                    **section,
                    "summary": section_summary,
                    "summary_length": len(section_summary)
                })
            
            return self._create_result(
                status=AgentStatus.COMPLETED,
                result_data={
                    "summarized_sections": summarized_sections,
                    "total_sections": len(summarized_sections),
                    "summary_method": "section_level"
                },
                metadata={
                    "agent_version": "1.0",
                    "processing_type": "section_summarization"
                }
            )
            
        except Exception as e:
            raise AgentError(f"Failed to summarize sections: {str(e)}", self.name)
    
    async def _summarize_section(self, section: Dict[str, Any], section_index: int) -> str:
        """Generate a summary for a single section."""
        content = section.get("content", "")
        title = section.get("title", f"Section {section_index + 1}")
        
        if not content.strip():
            return f"Empty section: {title}"
        
        # For now, implement a simple extractive summarization
        # In a real implementation, this would use an LLM
        sentences = content.split('. ')
        if len(sentences) <= 3:
            return content[:self.summary_length]
        
        # Take first few sentences as summary
        summary_sentences = sentences[:2]
        summary = '. '.join(summary_sentences)
        
        if len(summary) > self.summary_length:
            summary = summary[:self.summary_length] + "..."
        
        return summary


class DocumentSummarizationAgent(BaseAgent):
    """Agent for summarizing entire documents."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="document_summarizer",
            description="Summarizes entire documents",
            config=config
        )
        self.summary_length = self.config.get("summary_length", 500)
        self.include_key_points = self.config.get("include_key_points", True)
        self.max_key_points = self.config.get("max_key_points", 5)
    
    async def validate_input(self, document: Document, context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate that the document has content to summarize."""
        if not document.extra_data or "sections" not in document.extra_data:
            self.logger.warning(f"Document {document.id} has no sections to summarize")
            return False
        
        sections = document.extra_data["sections"]
        if not sections:
            self.logger.warning(f"Document {document.id} has empty sections")
            return False
        
        return True
    
    async def process(
        self, 
        document: Document, 
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResult:
        """Generate a comprehensive summary of the document."""
        try:
            sections = document.extra_data["sections"]
            
            # Combine all section content
            full_content = self._combine_sections(sections)
            
            # Generate document summary
            summary = await self._generate_document_summary(full_content, document)
            
            # Extract key points
            key_points = []
            if self.include_key_points:
                key_points = await self._extract_key_points(sections)
            
            # Include entities if available
            entities = document.extra_data.get("all_entities", [])
            entity_summary = self._summarize_entities(entities)
            
            return self._create_result(
                status=AgentStatus.COMPLETED,
                result_data={
                    "summary": summary,
                    "key_points": key_points,
                    "entity_summary": entity_summary,
                    "summary_length": len(summary),
                    "total_sections": len(sections),
                    "summary_method": "document_level"
                },
                metadata={
                    "agent_version": "1.0",
                    "processing_type": "document_summarization",
                    "include_entities": len(entities) > 0
                }
            )
            
        except Exception as e:
            raise AgentError(f"Failed to summarize document: {str(e)}", self.name)
    
    def _combine_sections(self, sections: List[Dict[str, Any]]) -> str:
        """Combine all section content into a single text."""
        combined = []
        for section in sections:
            title = section.get("title", "")
            content = section.get("content", "")
            if title:
                combined.append(f"## {title}")
            combined.append(content)
        return "\n\n".join(combined)
    
    async def _generate_document_summary(self, content: str, document: Document) -> str:
        """Generate a comprehensive document summary."""
        # For now, implement a simple extractive summarization
        # In a real implementation, this would use an LLM like GPT-4
        
        sentences = content.split('. ')
        if len(sentences) <= 5:
            return content[:self.summary_length]
        
        # Take first few sentences and some from the middle
        summary_sentences = sentences[:3]
        if len(sentences) > 6:
            mid_point = len(sentences) // 2
            summary_sentences.extend(sentences[mid_point:mid_point + 2])
        
        summary = '. '.join(summary_sentences)
        
        if len(summary) > self.summary_length:
            summary = summary[:self.summary_length] + "..."
        
        return summary
    
    async def _extract_key_points(self, sections: List[Dict[str, Any]]) -> List[str]:
        """Extract key points from the document sections."""
        key_points = []
        
        for section in sections:
            title = section.get("title", "")
            if title and len(key_points) < self.max_key_points:
                key_points.append(title)
        
        # If we need more key points, extract from content
        if len(key_points) < self.max_key_points:
            for section in sections:
                content = section.get("content", "")
                sentences = content.split('. ')
                for sentence in sentences[:2]:  # Take first 2 sentences
                    if len(sentence.strip()) > 20 and len(key_points) < self.max_key_points:
                        key_points.append(sentence.strip()[:100] + "...")
        
        return key_points[:self.max_key_points]
    
    def _summarize_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of extracted entities."""
        if not entities:
            return {"total_entities": 0, "entity_types": {}}
        
        entity_types = {}
        for entity in entities:
            entity_type = entity.get("type", "UNKNOWN")
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        return {
            "total_entities": len(entities),
            "entity_types": entity_types,
            "most_common_type": max(entity_types.items(), key=lambda x: x[1])[0] if entity_types else None
        }


class CorpusSummarizationAgent(BaseAgent):
    """Agent for summarizing collections of documents (corpus-level)."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="corpus_summarizer",
            description="Summarizes collections of documents",
            config=config
        )
        self.summary_length = self.config.get("summary_length", 1000)
        self.max_documents = self.config.get("max_documents", 50)
        self.include_trends = self.config.get("include_trends", True)
    
    async def validate_input(self, document: Document, context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate that we have a corpus of documents to summarize."""
        if not context or "corpus_documents" not in context:
            self.logger.warning("No corpus documents provided in context")
            return False
        
        corpus_documents = context["corpus_documents"]
        if not corpus_documents or len(corpus_documents) == 0:
            self.logger.warning("Empty corpus provided")
            return False
        
        return True
    
    async def process(
        self, 
        document: Document, 
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResult:
        """Generate a corpus-level summary."""
        try:
            corpus_documents = context["corpus_documents"]
            
            # Limit the number of documents to process
            if len(corpus_documents) > self.max_documents:
                corpus_documents = corpus_documents[:self.max_documents]
                self.logger.warning(f"Limited corpus to {self.max_documents} documents")
            
            # Analyze the corpus
            corpus_analysis = await self._analyze_corpus(corpus_documents)
            
            # Generate corpus summary
            corpus_summary = await self._generate_corpus_summary(corpus_analysis)
            
            # Identify trends if requested
            trends = []
            if self.include_trends:
                trends = await self._identify_trends(corpus_documents)
            
            return self._create_result(
                status=AgentStatus.COMPLETED,
                result_data={
                    "corpus_summary": corpus_summary,
                    "trends": trends,
                    "corpus_analysis": corpus_analysis,
                    "total_documents": len(corpus_documents),
                    "summary_method": "corpus_level"
                },
                metadata={
                    "agent_version": "1.0",
                    "processing_type": "corpus_summarization",
                    "max_documents_processed": len(corpus_documents)
                }
            )
            
        except Exception as e:
            raise AgentError(f"Failed to summarize corpus: {str(e)}", self.name)
    
    async def _analyze_corpus(self, documents: List[Document]) -> Dict[str, Any]:
        """Analyze the corpus of documents."""
        total_sections = 0
        total_entities = 0
        entity_types = {}
        document_types = {}
        
        for doc in documents:
            # Count sections
            if doc.extra_data and "sections" in doc.extra_data:
                total_sections += len(doc.extra_data["sections"])
            
            # Count entities
            if doc.extra_data and "all_entities" in doc.extra_data:
                entities = doc.extra_data["all_entities"]
                total_entities += len(entities)
                
                for entity in entities:
                    entity_type = entity.get("type", "UNKNOWN")
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            # Track document types
            doc_type = doc.content_type or "unknown"
            document_types[doc_type] = document_types.get(doc_type, 0) + 1
        
        return {
            "total_documents": len(documents),
            "total_sections": total_sections,
            "total_entities": total_entities,
            "entity_type_distribution": entity_types,
            "document_type_distribution": document_types,
            "average_sections_per_document": total_sections / len(documents) if documents else 0,
            "average_entities_per_document": total_entities / len(documents) if documents else 0
        }
    
    async def _generate_corpus_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate a summary of the corpus analysis."""
        summary_parts = []
        
        # Document count
        total_docs = analysis["total_documents"]
        summary_parts.append(f"This corpus contains {total_docs} documents.")
        
        # Sections
        total_sections = analysis["total_sections"]
        avg_sections = analysis["average_sections_per_document"]
        summary_parts.append(f"The documents contain a total of {total_sections} sections, with an average of {avg_sections:.1f} sections per document.")
        
        # Entities
        total_entities = analysis["total_entities"]
        avg_entities = analysis["average_entities_per_document"]
        summary_parts.append(f"A total of {total_entities} entities were extracted, averaging {avg_entities:.1f} entities per document.")
        
        # Most common entity type
        entity_dist = analysis["entity_type_distribution"]
        if entity_dist:
            most_common = max(entity_dist.items(), key=lambda x: x[1])
            summary_parts.append(f"The most common entity type is {most_common[0]} with {most_common[1]} occurrences.")
        
        # Document types
        doc_types = analysis["document_type_distribution"]
        if len(doc_types) > 1:
            type_summary = ", ".join([f"{doc_type} ({count})" for doc_type, count in doc_types.items()])
            summary_parts.append(f"Document types include: {type_summary}.")
        
        return " ".join(summary_parts)
    
    async def _identify_trends(self, documents: List[Document]) -> List[str]:
        """Identify trends across the corpus."""
        trends = []
        
        # Analyze entity trends
        entity_trends = self._analyze_entity_trends(documents)
        if entity_trends:
            trends.extend(entity_trends)
        
        # Analyze content trends
        content_trends = self._analyze_content_trends(documents)
        if content_trends:
            trends.extend(content_trends)
        
        return trends[:5]  # Limit to 5 trends
    
    def _analyze_entity_trends(self, documents: List[Document]) -> List[str]:
        """Analyze entity-related trends."""
        trends = []
        
        # Count entities by type across all documents
        entity_counts = {}
        for doc in documents:
            if doc.extra_data and "all_entities" in doc.extra_data:
                for entity in doc.extra_data["all_entities"]:
                    entity_type = entity.get("type", "UNKNOWN")
                    entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        # Identify dominant entity types
        if entity_counts:
            sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_entities) >= 2:
                top_entity = sorted_entities[0]
                second_entity = sorted_entities[1]
                if top_entity[1] > second_entity[1] * 1.5:  # Significantly more common
                    trends.append(f"{top_entity[0]} entities are the most prevalent in this corpus, appearing {top_entity[1]} times.")
        
        return trends
    
    def _analyze_content_trends(self, documents: List[Document]) -> List[str]:
        """Analyze content-related trends."""
        trends = []
        
        # Analyze section length trends
        section_lengths = []
        for doc in documents:
            if doc.extra_data and "sections" in doc.extra_data:
                for section in doc.extra_data["sections"]:
                    content = section.get("content", "")
                    section_lengths.append(len(content))
        
        if section_lengths:
            avg_length = sum(section_lengths) / len(section_lengths)
            if avg_length > 1000:
                trends.append("Documents in this corpus tend to have longer sections, with an average section length over 1000 characters.")
            elif avg_length < 200:
                trends.append("Documents in this corpus tend to have shorter sections, with an average section length under 200 characters.")
        
        return trends
