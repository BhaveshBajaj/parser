"""AutoGen agent for question answering tasks."""

import logging
from typing import Any, Dict, Optional

from .base_agent import BaseAutoGenAgent

logger = logging.getLogger(__name__)


class QAAgent(BaseAutoGenAgent):
    """AutoGen agent for question answering tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        system_message = """You are an expert question-answering agent for document analysis. Your role is to:

1. Answer questions accurately based on document content
2. Provide evidence and citations for answers
3. Handle both factual and analytical questions
4. Assess confidence in answers
5. Identify when questions cannot be answered from available content

CRITICAL: You MUST always respond with ONLY valid JSON. Never include explanations, markdown formatting, or any other text.

When answering questions:
- Base answers strictly on document content
- Provide specific evidence and quotes when possible
- Indicate confidence levels in your answers
- Clearly state when information is not available
- Consider context and relationships between concepts

Always respond with structured JSON containing:
- answer: Direct answer to the question
- evidence: Supporting evidence from the document
- confidence: Confidence score (0.0-1.0)
- sources: Specific sections or parts of document used
- related_entities: Relevant entities that support the answer

Remember: ONLY JSON output, no other text or formatting.
"""
        
        super().__init__(
            name="qa_agent",
            system_message=system_message,
            config=config
        )
    
    async def generate_qa_prompt(self, document_content: str, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a prompt for question answering using semantic search."""
        # Use semantic search to find most relevant content
        relevant_content = await self._find_relevant_content_semantic(question, document_content)
        
        context_info = ""
        if context:
            if "entities" in context:
                entities = context["entities"][:10]  # Limit to top 10
                entity_list = [f"{e.get('text', '')} ({e.get('type', '')})" for e in entities]
                context_info += f"\n\nRelevant entities: {', '.join(entity_list)}"
            
            if "section_summaries" in context:
                context_info += f"\n\nDocument has {len(context['section_summaries'])} sections available."
        
        # Add semantically relevant content
        if relevant_content:
            context_info += f"\n\nMost relevant content (found using semantic search):\n{relevant_content}"
        
        return f"""Answer question. Return ONLY JSON:

Question: {question}

Document:
{document_content}
{context_info}

{{
  "answer": "answer",
  "evidence": "evidence",
  "confidence": 0.95,
  "reasoning": "reasoning",
  "sources": "sources",
  "related_information": "info",
  "limitations": "limitations",
  "semantic_relevance": "relevance"
}}

ONLY JSON response."""
    
    async def _find_relevant_content_semantic(self, question: str, document_content: str) -> str:
        """Find relevant content using semantic embeddings."""
        try:
            # Split document into chunks for semantic search
            chunks = self.embedding_service.chunk_text(document_content)
            if not chunks:
                return ""
            
            # Get embeddings for question and chunks
            question_embedding = await self.embedding_service.embed_text(question)
            if not question_embedding:
                return ""
            
            # Find most similar chunks
            similarities = []
            for chunk in chunks:
                chunk_embedding = await self.embedding_service.embed_text(chunk['text'])
                if chunk_embedding:
                    similarity = self.embedding_service.cosine_similarity(question_embedding, chunk_embedding)
                    similarities.append((chunk['text'], similarity))
            
            # Sort by similarity and get top 3 chunks
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_chunks = [chunk_text for chunk_text, _ in similarities[:3]]
            
            return "\n\n".join(top_chunks)
            
        except Exception as e:
            logger.error(f"Semantic content search failed: {str(e)}")
            return ""
    
    def generate_corpus_qa_prompt(self, documents_content: str, question: str) -> str:
        """Generate a prompt for corpus-level question answering."""
        return f"""Please answer the following question based on multiple documents provided.

Question: {question}

Multiple documents content:
{documents_content}

Analyze across all documents and provide a JSON response with:
- answer: Comprehensive answer drawing from all documents
- evidence_by_document: Evidence organized by document
- confidence: Overall confidence in the answer
- cross_document_insights: Insights that emerge from comparing documents
- document_relevance: Which documents were most relevant to the question
- synthesis: How information from different documents relates
"""
