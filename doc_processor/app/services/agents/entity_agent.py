"""AutoGen agent for entity extraction and processing."""

import logging
from typing import Any, Dict, List, Optional

from .base_agent import BaseAutoGenAgent
from ...models.document import Document

logger = logging.getLogger(__name__)


class EntityAgent(BaseAutoGenAgent):
    """AutoGen agent for entity extraction and processing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        system_message = """You are an expert entity extraction and analysis agent. Your role is to:

1. Extract named entities from documents (people, organizations, locations, dates, etc.)
2. Categorize and tag entities appropriately
3. Validate entity quality and relevance
4. Identify relationships between entities
5. Provide confidence scores for extractions

CRITICAL: You MUST always respond with ONLY valid JSON. Never include explanations, markdown formatting, or any other text.

When processing documents:
- Extract all relevant named entities
- Classify entities by type (PERSON, ORG, GPE, DATE, MONEY, etc.)
- Provide confidence scores (0.0-1.0)
- Identify entity relationships and contexts
- Filter out low-quality or irrelevant entities

Always respond with structured JSON containing:
- entities: List of extracted entities with types and confidence
- entity_relationships: Relationships between entities
- entity_categories: Categorization of entities
- validation_results: Quality assessment of extractions

Remember: ONLY JSON output, no other text or formatting.
"""
        
        super().__init__(
            name="entity_agent",
            system_message=system_message,
            config=config
        )
    
    def generate_entity_prompt(self, document_content: str, entity_types: Optional[List[str]] = None) -> str:
        """Generate a prompt for entity extraction."""
        entity_types_str = ", ".join(entity_types) if entity_types else "all types"
        
        # Limit document content to prevent context window overflow
        max_content_length = 6000  # Leave room for prompt and response
        if len(document_content) > max_content_length:
            document_content = document_content[:max_content_length] + "..."
        
        return f"""Extract entities from this document. Return ONLY JSON:

Document:
{document_content}

JSON format:
{{
  "entities": [
    {{"text": "entity", "type": "PERSON|ORG|GPE|DATE|MONEY|PRODUCT|LOCATION|EVENT", "confidence": 0.95, "context": "context", "start_pos": 123, "end_pos": 135}}
  ],
  "entity_summary": {{"total_entities": 15, "by_type": {{"PERSON": 5, "ORG": 3}}, "high_confidence_count": 12}},
  "relationships": [{{"entity1": "name", "entity2": "org", "relationship": "works_at", "confidence": 0.9}}]
}}

Extract: people, organizations, locations, dates, products, events, money amounts. ONLY JSON response."""
    
    async def extract_entities_with_llm(self, document: Document) -> Dict[str, Any]:
        """LLM-powered entity extraction using Azure OpenAI."""
        try:
            # Get document content
            document_content = self.get_document_content()
            if not document_content:
                logger.warning("No document content available for entity extraction")
                return {
                    "entities": [],
                    "total_entities": 0,
                    "extraction_method": "no_content",
                    "error": "No document content available"
                }
            
            # Generate entity extraction prompt
            prompt = self.generate_entity_prompt(document_content)
            
            # Use the agent's LLM to extract entities
            import asyncio
            import autogen
            
            # Create a simple user proxy for the agent interaction
            user_proxy = autogen.UserProxyAgent(
                name="entity_extractor_user",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                code_execution_config=False,
            )
            
            # Get response from the entity agent
            response = await asyncio.to_thread(
                user_proxy.initiate_chat,
                self,
                message=prompt
            )
            
            # Extract the response content
            response_content = ""
            if hasattr(response, 'chat_history') and response.chat_history:
                last_message = response.chat_history[-1]
                if isinstance(last_message, dict) and 'content' in last_message:
                    response_content = last_message['content']
            
            # Try to parse JSON response
            import json
            try:
                parsed_response = json.loads(response_content)
                entities_data = parsed_response.get("entities", [])
                
                logger.info(f"Successfully extracted {len(entities_data)} entities using LLM")
                return {
                    "entities": entities_data,
                    "total_entities": len(entities_data),
                    "extraction_method": "azure_openai_llm",
                    "model_used": "Azure OpenAI",
                    "raw_response": response_content
                }
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response from entity agent: {response_content}")
                return {
                    "entities": [],
                    "total_entities": 0,
                    "extraction_method": "json_parse_failed",
                    "error": "Failed to parse JSON response",
                    "raw_response": response_content
                }
                
        except Exception as e:
            logger.error(f"LLM entity extraction failed: {str(e)}", exc_info=True)
            return {
                "entities": [],
                "total_entities": 0,
                "extraction_method": "llm_failed",
                "error": str(e)
            }
