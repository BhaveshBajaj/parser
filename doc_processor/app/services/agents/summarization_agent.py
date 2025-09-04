"""AutoGen agent for document summarization tasks."""

from typing import Any, Dict, Optional

from .base_agent import BaseAutoGenAgent


class SummarizationAgent(BaseAutoGenAgent):
    """AutoGen agent for document summarization tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        system_message = """You are an expert document summarization agent. Your role is to:

1. Analyze document content and create concise, accurate summaries
2. Extract key points and main themes
3. Identify important entities and concepts
4. Provide section-level and document-level summaries
5. Ensure summaries are factual and preserve important details

CRITICAL: You MUST always respond with ONLY valid JSON. Never include explanations, markdown formatting, or any other text.

When processing documents:
- Focus on the main ideas and key information
- Maintain factual accuracy
- Use clear, professional language
- Structure summaries logically
- Include relevant entities and dates when present

Always respond with structured JSON containing:
- summary: Main document summary
- key_points: List of key points
- entities_mentioned: Important entities found
- section_summaries: Summaries of individual sections if applicable

Remember: ONLY JSON output, no other text or formatting.
"""
        
        super().__init__(
            name="summarization_agent",
            system_message=system_message,
            config=config
        )
    
    def generate_summary_prompt(self, document_content: str, summary_type: str = "document") -> str:
        """Generate a prompt for summarization based on the type."""
        if summary_type == "section":
            return f"""Summarize this section. Return ONLY JSON:

{document_content}

{{
  "section_summary": "summary",
  "key_points": ["point1", "point2"],
  "entities": ["entity1", "entity2"]
}}

ONLY JSON response."""
        elif summary_type == "corpus":
            return f"""Summarize documents. Return ONLY JSON:

{document_content}

{{
  "corpus_summary": "summary",
  "common_themes": ["theme1"],
  "key_entities": ["entity1"],
  "document_relationships": "relationships"
}}

ONLY JSON response."""
        else:  # document level
            return f"""Summarize document. Return ONLY JSON:

{document_content}

{{
  "summary": "summary",
  "key_points": ["point1", "point2", "point3"],
  "entities_mentioned": ["entity1", "entity2"],
  "main_topics": ["topic1"],
  "document_type": "type"
}}

ONLY JSON response."""
