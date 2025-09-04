"""Base AutoGen agent for document processing tasks."""

import logging
from typing import Any, Dict, Optional

import autogen
from autogen import ConversableAgent

from ...models.document import Document
from ..embedding_service import get_embedding_service
from ...core.config import settings

logger = logging.getLogger(__name__)


class BaseAutoGenAgent(ConversableAgent):
    """Base AutoGen agent for document processing tasks."""
    
    def __init__(
        self,
        name: str,
        system_message: str,
        llm_config: Optional[Dict] = None,
        human_input_mode: str = "NEVER",
        max_consecutive_auto_reply: int = 3,
        config: Optional[Dict] = None,
        **kwargs
    ):
        # Configure LLM settings
        if llm_config is None and settings.AZURE_OPENAI_API_KEY:
            # Use the deployment name from settings to avoid hardcoding
            model_name = getattr(settings, 'AZURE_OPENAI_DEPLOYMENT', 'gpt-4o')
            llm_config = {
                "config_list": [{
                    "model": model_name,
                    "api_key": settings.AZURE_OPENAI_API_KEY,
                    "base_url": settings.AZURE_OPENAI_ENDPOINT,
                    "api_type": "azure",
                    "api_version": settings.AZURE_OPENAI_API_VERSION,
                }],
                "temperature": 0.3,
                "timeout": 120,
            }
        
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode=human_input_mode,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            **kwargs
        )
        
        # Store document processing context
        self.document_context: Optional[Dict[str, Any]] = None
        self.processing_results: Dict[str, Any] = {}
        
        # Initialize embedding service
        self.config = config or {}
        self.embedding_service = get_embedding_service(self.config)
    
    def set_document_context(self, document: Document, context: Optional[Dict[str, Any]] = None):
        """Set the document and context for processing."""
        self.document_context = {
            "document": document,
            "context": context or {},
            "document_id": str(document.id),
            "filename": document.filename,
            "content_type": document.content_type
        }
    
    def get_document_content(self, max_length: int = 8000) -> str:
        """Extract readable content from the document with length limits."""
        if not self.document_context:
            return ""
        
        document = self.document_context["document"]
        
        # Get content from sections if available
        if document.extra_data and "sections" in document.extra_data:
            sections = document.extra_data["sections"]
            content_parts = []
            total_length = 0
            
            for section in sections:
                title = section.get("title", "")
                content = section.get("content", "")
                
                section_text = ""
                if title:
                    section_text = f"## {title}\n"
                section_text += content
                
                # Check if adding this section would exceed max_length
                if total_length + len(section_text) > max_length:
                    # Truncate the last section if needed
                    remaining_space = max_length - total_length
                    if remaining_space > 100:  # Only add if there's meaningful space
                        section_text = section_text[:remaining_space] + "..."
                        content_parts.append(section_text)
                    break
                
                content_parts.append(section_text)
                total_length += len(section_text)
            
            return "\n\n".join(content_parts)
        
        # Use summary or return filename if no content available
        if document.summary:
            return document.summary[:max_length]
        
        return f"Document: {document.filename} (Content not available in processed format)"
