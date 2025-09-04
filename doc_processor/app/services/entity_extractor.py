"""Entity extraction service using Azure OpenAI."""
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import re
from dataclasses import dataclass, asdict
from enum import Enum

import openai
from openai import AzureOpenAI
from pydantic import BaseModel, Field

from ..core.config import settings

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Supported entity types for extraction."""
    PERSON = "PERSON"
    ORG = "ORG"
    GPE = "GPE"  # Geo-Political Entity (countries, cities, states)
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    NORP = "NORP"  # Nationalities, religious, or political groups
    FAC = "FAC"    # Buildings, airports, etc.
    LOC = "LOC"    # Non-GPE locations
    LOCATION = "LOCATION"  # General location entities (alias for LOC)
    WORK_OF_ART = "WORK_OF_ART"
    LAW = "LAW"
    LANGUAGE = "LANGUAGE"
    QUANTITY = "QUANTITY"
    ORDINAL = "ORDINAL"
    CARDINAL = "CARDINAL"


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    type: EntityType
    start: int
    end: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

    def to_dict(self):
        """Convert to dictionary."""
        result = asdict(self)
        result["type"] = self.type.value
        return result


class EntityExtractor:
    """Service for extracting entities from text using Azure OpenAI."""
    
    def __init__(self):
        """Initialize the entity extractor with Azure OpenAI client."""
        if not settings.AZURE_OPENAI_API_KEY or not settings.AZURE_OPENAI_ENDPOINT:
            logger.warning("Azure OpenAI credentials not configured. Entity extraction will be disabled.")
            self.client = None
            self.deployment = None
        else:
            self.client = AzureOpenAI(
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
            )
            self.deployment = settings.AZURE_OPENAI_DEPLOYMENT
        
    async def extract_entities(
        self, 
        text: str, 
        entity_types: Optional[List[EntityType]] = None,
        **kwargs
    ) -> List[Entity]:
        """
        Extract entities from the given text using Azure OpenAI.
        
        Args:
            text: The text to extract entities from
            entity_types: List of entity types to extract. If None, uses default from settings.
            **kwargs: Additional arguments to pass to the OpenAI API
            
        Returns:
            List of extracted entities
        """
        if not text.strip():
            return []
            
        if entity_types is None:
            entity_types = [EntityType(t) for t in settings.ENTITY_TYPES]
        
        # If Azure OpenAI is not configured, return empty list
        if not self.client or not self.deployment:
            logger.info("Azure OpenAI not configured, returning empty entities list")
            return []
            
        # Prepare the system message
        system_message = self._create_system_message(entity_types)
        
        try:
            # Call Azure OpenAI API (synchronous call, not async)
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": text}
                ],
                temperature=0.0,
                **kwargs
            )
            
            # Parse the response
            content = response.choices[0].message.content
            entities = self._parse_response(content, text)
            
            # Validate and filter entities
            entities = self._validate_entities(entities, text)
            
            logger.info(f"Extracted {len(entities)} entities")
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            # Return empty list on error instead of mock entities
            logger.info("Returning empty entities list due to API error")
            return []
    
    def _create_system_message(self, entity_types: List[EntityType]) -> str:
        """Create the system message for entity extraction."""
        entity_descriptions = {
            EntityType.PERSON: "People, including fictional characters, proper names",
            EntityType.ORG: "Companies, agencies, institutions, organizations, corporations",
            EntityType.GPE: "Countries, cities, states, provinces, nations, geopolitical entities",
            EntityType.DATE: "Absolute or relative dates, periods, time references",
            EntityType.TIME: "Times smaller than a day (hours, minutes, seconds)",
            EntityType.MONEY: "Monetary values, currency amounts, financial figures",
            EntityType.PERCENT: "Percentage values, ratios, proportions",
            EntityType.PRODUCT: "Objects, vehicles, foods, physical products (not services)",
            EntityType.EVENT: "Named hurricanes, battles, wars, sports events, conferences",
            EntityType.NORP: "Nationalities, religious groups, political groups, ethnicities",
            EntityType.FAC: "Buildings, airports, highways, bridges, infrastructure",
            EntityType.LOC: "Non-GPE locations, mountain ranges, bodies of water, landmarks",
            EntityType.LOCATION: "General location entities, gates, stations, venues",
            EntityType.WORK_OF_ART: "Titles of books, songs, movies, artworks, creative works",
            EntityType.LAW: "Named documents made into laws, legal documents",
            EntityType.LANGUAGE: "Any named language, programming languages",
            EntityType.QUANTITY: "Measurements, counts, distances, weights, volumes",
            EntityType.ORDINAL: "First, second, third, etc. (ordinal numbers)",
            EntityType.CARDINAL: "Numerals that do not fall under another type",
        }
        
        # Filter entity types to only those requested
        entity_descriptions = {
            k: v for k, v in entity_descriptions.items() 
            if k in entity_types
        }
        
        # Format the entity types and descriptions
        entity_list = "\n".join(
            f"- {et.value}: {desc}" 
            for et, desc in entity_descriptions.items()
        )
        
        return f"""You are an expert named entity recognition system. Extract entities from the provided text and return them as a valid JSON array.

CRITICAL REQUIREMENTS:
1. Return ONLY valid JSON - no markdown, no explanations, no additional text
2. Extract entities that are clearly identifiable and contextually relevant
3. Provide EXACT character positions by counting from the beginning of the text
4. Use conservative confidence scores (0.7+ for high confidence, 0.5+ for medium)
5. Focus on meaningful entities that add value to document understanding

Entity types to extract:
{entity_list}

EXTRACTION RULES:
- PERSON: Full names (first + last), not just first names or titles
- ORG: Complete organization names, not abbreviations unless primary identifier
- GPE: Specific geographic locations, not general terms like "city" or "country"
- DATE: Specific dates, years, or time periods, not relative terms like "today"
- MONEY: Monetary amounts with currency symbols or words
- Only extract entities that appear in the text exactly as written
- Skip entities shorter than 2 characters or too generic

REQUIRED JSON FORMAT:
[
  {{
    "text": "exact entity text from document",
    "type": "ENTITY_TYPE",
    "start": 0,
    "end": 10,
    "confidence": 0.85,
    "metadata": {{}}
  }}
]

IMPORTANT: Return ONLY the JSON array. No markdown code blocks, no explanations, no additional text."""
    
    def _parse_response(self, response_text: str, original_text: str) -> List[Entity]:
        """Parse the response from the model into Entity objects."""
        try:
            # Clean the response text to extract JSON
            json_str = self._extract_json_from_response(response_text)
            entities_data = json.loads(json_str)
            
            if not isinstance(entities_data, list):
                raise ValueError("Expected a list of entities")
                
            entities = []
            for item in entities_data:
                try:
                    # Validate and create entity
                    entity = self._create_entity_from_dict(item, original_text)
                    if entity:
                        entities.append(entity)
                except Exception as e:
                    logger.warning(f"Skipping invalid entity: {item}, error: {str(e)}")
                    continue
                    
            return entities
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse entity extraction response: {response_text[:200]}...")
            raise ValueError(f"Invalid JSON response from model: {str(e)}")
        except Exception as e:
            logger.error(f"Error parsing entities: {str(e)}")
            raise
    
    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from the LLM response, handling various formats."""
        # Remove any markdown code blocks
        json_match = re.search(r'```(?:json\n)?([\s\S]*?)\n```', response_text)
        if json_match:
            return json_match.group(1).strip()
        
        # Look for JSON array pattern
        json_match = re.search(r'\[[\s\S]*\]', response_text)
        if json_match:
            return json_match.group(0).strip()
        
        # Try to parse the whole response as JSON
        return response_text.strip()
    
    def _create_entity_from_dict(self, item: dict, original_text: str) -> Optional[Entity]:
        """Create an Entity object from a dictionary, with validation."""
        try:
            # Validate required fields
            if not all(key in item for key in ["text", "type", "start", "end"]):
                logger.warning(f"Missing required fields in entity: {item}")
                return None
            
            # Validate entity type
            entity_type = EntityType(item["type"].upper())
            
            # Create entity
            entity = Entity(
                text=item["text"],
                type=entity_type,
                start=int(item["start"]),
                end=int(item["end"]),
                confidence=float(item.get("confidence", 0.5)),
                metadata=item.get("metadata", {})
            )
            
            # Validate position and text match
            if not self._validate_entity_position(entity, original_text):
                return None
                
            return entity
            
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Invalid entity data: {item}, error: {str(e)}")
            return None
    
    def _validate_entity_position(self, entity: Entity, original_text: str) -> bool:
        """Validate that entity position and text match the original text."""
        # Check bounds
        if entity.start < 0 or entity.end > len(original_text) or entity.start >= entity.end:
            logger.warning(f"Entity position out of bounds: start={entity.start}, end={entity.end}, text_length={len(original_text)}")
            return False
        
        # Check text match
        actual_text = original_text[entity.start:entity.end]
        if actual_text != entity.text:
            logger.warning(f"Entity text mismatch: '{entity.text}' != '{actual_text}' (start={entity.start}, end={entity.end})")
            # Try to find correct position
            corrected = self._find_correct_position(entity.text, original_text, entity.start)
            if corrected:
                entity.start, entity.end = corrected
                logger.info(f"Found correct position at {entity.start}-{entity.end}")
                return True
            else:
                return False
        
        return True
    
    def _validate_entities(self, entities: List[Entity], original_text: str) -> List[Entity]:
        """Validate and filter entities based on quality criteria."""
        if not entities:
            return []
        
        validated_entities = []
        
        for entity in entities:
            # Skip entities that are too short
            if len(entity.text.strip()) < 2:
                logger.debug(f"Skipping entity too short: '{entity.text}'")
                continue
            
            # Skip entities with very low confidence
            if entity.confidence < 0.3:
                logger.debug(f"Skipping entity with low confidence: '{entity.text}' (confidence: {entity.confidence})")
                continue
            
            # Skip common words and articles
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            if entity.text.lower().strip() in common_words:
                logger.debug(f"Skipping common word: '{entity.text}'")
                continue
            
            # Skip entities that are just numbers (unless they're meaningful)
            if entity.text.strip().isdigit() and entity.type not in [EntityType.CARDINAL, EntityType.ORDINAL, EntityType.QUANTITY]:
                logger.debug(f"Skipping numeric entity: '{entity.text}'")
                continue
            
            # Skip entities that are just punctuation
            if not any(c.isalnum() for c in entity.text):
                logger.debug(f"Skipping non-alphanumeric entity: '{entity.text}'")
                continue
            
            validated_entities.append(entity)
        
        logger.info(f"Validated {len(validated_entities)}/{len(entities)} entities")
        return validated_entities
    
    def _find_correct_position(self, entity_text: str, original_text: str, hint_start: int = 0) -> Optional[Tuple[int, int]]:
        """
        Find the correct position of an entity in the original text using multiple strategies.
        
        Args:
            entity_text: The entity text to find
            original_text: The original text to search in
            hint_start: A hint for where to start searching
            
        Returns:
            Tuple of (start, end) positions if found, None otherwise
        """
        if not entity_text or not original_text:
            return None
            
        # Strategy 1: Exact match (case-sensitive)
        start_pos = original_text.find(entity_text, hint_start)
        if start_pos != -1:
            return (start_pos, start_pos + len(entity_text))
        
        # Strategy 2: Case-insensitive match
        text_lower = original_text.lower()
        entity_lower = entity_text.lower()
        start_pos = text_lower.find(entity_lower, hint_start)
        if start_pos != -1:
            return (start_pos, start_pos + len(entity_text))
        
        # Strategy 3: Search from the beginning if hint didn't work
        if hint_start > 0:
            start_pos = original_text.find(entity_text, 0)
            if start_pos != -1:
                return (start_pos, start_pos + len(entity_text))
            
            start_pos = text_lower.find(entity_lower, 0)
            if start_pos != -1:
                return (start_pos, start_pos + len(entity_text))
        
        # Strategy 4: Handle common formatting issues
        # Remove extra whitespace and normalize
        normalized_entity = ' '.join(entity_text.split())
        if normalized_entity != entity_text:
            start_pos = original_text.find(normalized_entity, hint_start)
            if start_pos != -1:
                return (start_pos, start_pos + len(normalized_entity))
            
            start_pos = text_lower.find(normalized_entity.lower(), hint_start)
            if start_pos != -1:
                return (start_pos, start_pos + len(normalized_entity))
        
        # Strategy 5: Partial matching for common issues
        # Handle cases where the entity might be split across lines or have formatting issues
        entity_clean = re.sub(r'[^\w\s]', '', entity_text.lower())
        text_clean = re.sub(r'[^\w\s]', '', original_text.lower())
        
        if entity_clean in text_clean:
            start_pos = text_clean.find(entity_clean, hint_start)
            if start_pos != -1:
                # Map back to original text position
                # Count non-alphanumeric characters before the match
                char_count = 0
                for i, char in enumerate(original_text):
                    if char_count == start_pos:
                        return (i, i + len(entity_text))
                    if char.isalnum() or char.isspace():
                        char_count += 1
        
        # Strategy 6: Try without punctuation at the end
        if entity_text and not entity_text[-1].isalnum():
            entity_no_punct = entity_text.rstrip('.,!?;:')
            if entity_no_punct != entity_text:
                start_pos = original_text.find(entity_no_punct, hint_start)
                if start_pos != -1:
                    return (start_pos, start_pos + len(entity_no_punct))
        
        return None
    
    async def extract_entities_batch(
        self, 
        texts: List[str], 
        entity_types: Optional[List[EntityType]] = None,
        batch_size: int = 10,
        **kwargs
    ) -> List[List[Entity]]:
        """
        Extract entities from multiple texts in batches.
        
        Args:
            texts: List of texts to extract entities from
            entity_types: List of entity types to extract
            batch_size: Number of texts to process in parallel
            **kwargs: Additional arguments to pass to the OpenAI API
            
        Returns:
            List of lists of entities, one list per input text
        """
        # TODO: Implement batching for better performance
        results = []
        for text in texts:
            entities = await self.extract_entities(text, entity_types, **kwargs)
            results.append(entities)
        return results
    

