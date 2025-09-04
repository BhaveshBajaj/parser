"""Entity models for document processing."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


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
    MISC = "MISC"  # Miscellaneous entities


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    type: EntityType
    start: int
    end: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "text": self.text,
            "type": self.type.value,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
