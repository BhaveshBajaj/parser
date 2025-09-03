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
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            # Return empty list on error instead of mock entities
            logger.info("Returning empty entities list due to API error")
            return []
    
    def _create_system_message(self, entity_types: List[EntityType]) -> str:
        """Create the system message for entity extraction."""
        entity_descriptions = {
            EntityType.PERSON: "People, including fictional",
            EntityType.ORG: "Companies, agencies, institutions, etc.",
            EntityType.GPE: "Countries, cities, states",
            EntityType.DATE: "Absolute or relative dates or periods",
            EntityType.TIME: "Times smaller than a day",
            EntityType.MONEY: "Monetary values, including unit",
            EntityType.PERCENT: "Percentage (including '%')",
            EntityType.PRODUCT: "Objects, vehicles, foods, etc. (not services)",
            EntityType.EVENT: "Named hurricanes, battles, wars, sports events, etc.",
            EntityType.NORP: "Nationalities or religious or political groups",
            EntityType.FAC: "Buildings, airports, highways, bridges, etc.",
            EntityType.LOC: "Non-GPE locations, mountain ranges, bodies of water",
            EntityType.LOCATION: "General location entities, gates, stations, etc.",
            EntityType.WORK_OF_ART: "Titles of books, songs, etc.",
            EntityType.LAW: "Named documents made into laws",
            EntityType.LANGUAGE: "Any named language",
            EntityType.QUANTITY: "Measurements, counts, distances, etc.",
            EntityType.ORDINAL: "First, second, etc.",
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
        
        return f"""You are a highly accurate named entity recognition system. 
Extract entities from the provided text and return them in the specified JSON format.

Entity types to extract (with descriptions):
{entity_list}

For each entity, provide:
- text: The exact text of the entity
- type: The entity type (must be one of the types listed above)
- start: The starting character offset of the entity in the original text
- end: The ending character offset (exclusive) of the entity in the original text
- confidence: A confidence score between 0 and 1
- metadata: Any additional metadata about the entity

Return the entities as a JSON array of objects with the above fields.
"""
    
    def _parse_response(self, response_text: str, original_text: str) -> List[Entity]:
        """Parse the response from the model into Entity objects."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'```(?:json\n)?([\s\S]*?)\n```', response_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no code block, try to parse the whole response as JSON
                json_str = response_text
                
            entities_data = json.loads(json_str)
            if not isinstance(entities_data, list):
                raise ValueError("Expected a list of entities")
                
            entities = []
            for item in entities_data:
                try:
                    # Validate entity type
                    entity_type = EntityType(item["type"].upper())
                    
                    # Create entity
                    entity = Entity(
                        text=item["text"],
                        type=entity_type,
                        start=item["start"],
                        end=item["end"],
                        confidence=item.get("confidence", 1.0),
                        metadata=item.get("metadata", {})
                    )
                    
                    # Verify the entity text matches the original text
                    if entity.start < len(original_text) and entity.end <= len(original_text):
                        actual_text = original_text[entity.start:entity.end]
                        if actual_text != entity.text:
                            logger.warning(
                                f"Entity text mismatch: '{entity.text}' != "
                                f"'{actual_text}' (start={entity.start}, end={entity.end})"
                            )
                            # Try to find the correct position using multiple strategies
                            corrected = self._find_correct_position(entity.text, original_text, entity.start)
                            if corrected:
                                entity.start, entity.end = corrected
                                logger.info(f"Found correct position at {entity.start}-{entity.end}")
                            else:
                                # If we can't find the correct position, try to find it anywhere in the text
                                corrected = self._find_correct_position(entity.text, original_text, 0)
                                if corrected:
                                    entity.start, entity.end = corrected
                                    logger.info(f"Found correct position at {entity.start}-{entity.end} (searched from beginning)")
                                else:
                                    logger.warning(f"Could not find correct position for entity: '{entity.text}' - skipping")
                                    continue  # Skip this entity if we can't find it
                    else:
                        logger.warning(
                            f"Entity position out of bounds: start={entity.start}, end={entity.end}, "
                            f"text_length={len(original_text)}"
                        )
                        # Try to find the correct position even if out of bounds
                        corrected = self._find_correct_position(entity.text, original_text, 0)
                        if corrected:
                            entity.start, entity.end = corrected
                            logger.info(f"Found correct position at {entity.start}-{entity.end}")
                        else:
                            logger.warning(f"Could not find correct position for out-of-bounds entity: '{entity.text}' - skipping")
                            continue  # Skip this entity if we can't find it
                        
                    entities.append(entity)
                    
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid entity: {item}, error: {str(e)}")
                    continue
                    
            return entities
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse entity extraction response: {response_text}")
            raise ValueError(f"Invalid JSON response from model: {str(e)}")
        except Exception as e:
            logger.error(f"Error parsing entities: {str(e)}")
            raise
    
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
    

