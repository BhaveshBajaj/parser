"""Entity extraction and tagging agents."""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from .base_agent import BaseAgent, AgentResult, AgentStatus, AgentError
from ...models.document import Document
from ..entity_extractor import Entity, EntityType, EntityExtractor

logger = logging.getLogger(__name__)


class EntityExtractionAgent(BaseAgent):
    """Agent for extracting entities from document content."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="entity_extractor",
            description="Extracts named entities from document content",
            config=config
        )
        self.entity_types = self.config.get("entity_types", None)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        self.extractor = EntityExtractor()
    
    async def validate_input(self, document: Document, context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate that the document has content to extract entities from."""
        if not document.extra_data or "sections" not in document.extra_data:
            self.logger.warning(f"Document {document.id} has no sections for entity extraction")
            return False
        
        sections = document.extra_data["sections"]
        if not sections:
            self.logger.warning(f"Document {document.id} has empty sections")
            return False
        
        # Check if at least one section has content
        has_content = any(section.get("content", "").strip() for section in sections)
        if not has_content:
            self.logger.warning(f"Document {document.id} has no content in sections")
            return False
        
        return True
    
    async def process(
        self, 
        document: Document, 
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResult:
        """Extract entities from all sections of the document."""
        try:
            sections = document.extra_data["sections"]
            all_entities = []
            section_entities = []
            
            for i, section in enumerate(sections):
                content = section.get("content", "")
                if not content.strip():
                    section_entities.append([])
                    continue
                
                # Extract entities from this section
                entities = await self.extractor.extract_entities(
                    text=content,
                    entity_types=self.entity_types
                )
                
                # Filter by confidence threshold
                filtered_entities = [
                    entity for entity in entities 
                    if entity.confidence >= self.confidence_threshold
                ]
                
                # Convert to dict format for storage
                entity_dicts = [entity.to_dict() for entity in filtered_entities]
                section_entities.append(entity_dicts)
                all_entities.extend(entity_dicts)
            
            # Analyze entity distribution
            entity_analysis = self._analyze_entities(all_entities)
            
            return self._create_result(
                status=AgentStatus.COMPLETED,
                result_data={
                    "all_entities": all_entities,
                    "section_entities": section_entities,
                    "entity_analysis": entity_analysis,
                    "total_entities": len(all_entities),
                    "confidence_threshold": self.confidence_threshold
                },
                metadata={
                    "agent_version": "1.0",
                    "processing_type": "entity_extraction",
                    "entity_types_requested": [t.value for t in (self.entity_types or [])],
                    "sections_processed": len(sections)
                }
            )
            
        except Exception as e:
            raise AgentError(f"Failed to extract entities: {str(e)}", self.name)
    
    def _analyze_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the extracted entities."""
        if not entities:
            return {"total_entities": 0, "entity_types": {}}
        
        entity_types = {}
        confidence_scores = []
        
        for entity in entities:
            entity_type = entity.get("type", "UNKNOWN")
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            confidence_scores.append(entity.get("confidence", 1.0))
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return {
            "total_entities": len(entities),
            "entity_types": entity_types,
            "average_confidence": avg_confidence,
            "most_common_type": max(entity_types.items(), key=lambda x: x[1])[0] if entity_types else None,
            "unique_entities": len(set(entity.get("text", "") for entity in entities))
        }


class EntityTaggingAgent(BaseAgent):
    """Agent for tagging and categorizing extracted entities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="entity_tagger",
            description="Tags and categorizes extracted entities",
            config=config
        )
        self.tag_categories = self.config.get("tag_categories", {
            "PERSON": "people",
            "ORG": "organizations", 
            "GPE": "locations",
            "DATE": "temporal",
            "MONEY": "financial",
            "PRODUCT": "products"
        })
        self.custom_tags = self.config.get("custom_tags", {})
    
    async def validate_input(self, document: Document, context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate that we have entities to tag."""
        if not context or "entities" not in context:
            self.logger.warning("No entities provided in context for tagging")
            return False
        
        entities = context["entities"]
        if not entities:
            self.logger.warning("Empty entities list provided for tagging")
            return False
        
        return True
    
    async def process(
        self, 
        document: Document, 
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResult:
        """Tag and categorize the extracted entities."""
        try:
            entities = context["entities"]
            tagged_entities = []
            
            for entity in entities:
                tagged_entity = await self._tag_entity(entity)
                tagged_entities.append(tagged_entity)
            
            # Analyze tag distribution
            tag_analysis = self._analyze_tags(tagged_entities)
            
            return self._create_result(
                status=AgentStatus.COMPLETED,
                result_data={
                    "tagged_entities": tagged_entities,
                    "tag_analysis": tag_analysis,
                    "total_tagged": len(tagged_entities)
                },
                metadata={
                    "agent_version": "1.0",
                    "processing_type": "entity_tagging",
                    "tag_categories_used": list(self.tag_categories.keys())
                }
            )
            
        except Exception as e:
            raise AgentError(f"Failed to tag entities: {str(e)}", self.name)
    
    async def _tag_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Tag a single entity with categories and custom tags."""
        entity_type = entity.get("type", "UNKNOWN")
        entity_text = entity.get("text", "")
        
        # Get category tag
        category_tag = self.tag_categories.get(entity_type, "other")
        
        # Apply custom tags based on entity text or type
        custom_tags = []
        for pattern, tag in self.custom_tags.items():
            if pattern.lower() in entity_text.lower():
                custom_tags.append(tag)
        
        # Add semantic tags based on entity characteristics
        semantic_tags = self._generate_semantic_tags(entity)
        
        tagged_entity = {
            **entity,
            "category_tag": category_tag,
            "custom_tags": custom_tags,
            "semantic_tags": semantic_tags,
            "all_tags": [category_tag] + custom_tags + semantic_tags
        }
        
        return tagged_entity
    
    def _generate_semantic_tags(self, entity: Dict[str, Any]) -> List[str]:
        """Generate semantic tags based on entity characteristics."""
        tags = []
        entity_text = entity.get("text", "").lower()
        entity_type = entity.get("type", "")
        
        # Length-based tags
        if len(entity_text) > 20:
            tags.append("long_entity")
        elif len(entity_text) < 3:
            tags.append("short_entity")
        
        # Content-based tags
        if any(char.isdigit() for char in entity_text):
            tags.append("contains_numbers")
        
        if any(char in entity_text for char in ["&", "and", "of", "the"]):
            tags.append("compound_entity")
        
        # Type-specific tags
        if entity_type in ["PERSON", "ORG"]:
            if entity_text.isupper():
                tags.append("formal_style")
        
        if entity_type == "DATE":
            if any(word in entity_text for word in ["january", "february", "march", "april", "may", "june", 
                                                   "july", "august", "september", "october", "november", "december"]):
                tags.append("month_mentioned")
        
        return tags
    
    def _analyze_tags(self, tagged_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the distribution of tags."""
        category_counts = {}
        custom_tag_counts = {}
        semantic_tag_counts = {}
        
        for entity in tagged_entities:
            # Count category tags
            category = entity.get("category_tag", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Count custom tags
            for tag in entity.get("custom_tags", []):
                custom_tag_counts[tag] = custom_tag_counts.get(tag, 0) + 1
            
            # Count semantic tags
            for tag in entity.get("semantic_tags", []):
                semantic_tag_counts[tag] = semantic_tag_counts.get(tag, 0) + 1
        
        return {
            "category_distribution": category_counts,
            "custom_tag_distribution": custom_tag_counts,
            "semantic_tag_distribution": semantic_tag_counts,
            "total_categories": len(category_counts),
            "total_custom_tags": len(custom_tag_counts),
            "total_semantic_tags": len(semantic_tag_counts)
        }


class EntityValidationAgent(BaseAgent):
    """Agent for validating and quality-checking extracted entities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="entity_validator",
            description="Validates and quality-checks extracted entities",
            config=config
        )
        self.min_confidence = self.config.get("min_confidence", 0.7)
        self.min_length = self.config.get("min_length", 2)
        self.max_length = self.config.get("max_length", 100)
        self.blacklist_words = self.config.get("blacklist_words", ["the", "and", "or", "but", "in", "on", "at"])
    
    async def validate_input(self, document: Document, context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate that we have entities to validate."""
        if not context or "entities" not in context:
            self.logger.warning("No entities provided in context for validation")
            return False
        
        entities = context["entities"]
        if not entities:
            self.logger.warning("Empty entities list provided for validation")
            return False
        
        return True
    
    async def process(
        self, 
        document: Document, 
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResult:
        """Validate the quality of extracted entities."""
        try:
            entities = context["entities"]
            validation_results = []
            valid_entities = []
            invalid_entities = []
            
            for entity in entities:
                validation_result = await self._validate_entity(entity)
                validation_results.append(validation_result)
                
                if validation_result["is_valid"]:
                    valid_entities.append(entity)
                else:
                    invalid_entities.append(entity)
            
            # Generate validation summary
            validation_summary = self._generate_validation_summary(validation_results)
            
            return self._create_result(
                status=AgentStatus.COMPLETED,
                result_data={
                    "validation_results": validation_results,
                    "valid_entities": valid_entities,
                    "invalid_entities": invalid_entities,
                    "validation_summary": validation_summary,
                    "total_entities": len(entities),
                    "valid_count": len(valid_entities),
                    "invalid_count": len(invalid_entities)
                },
                metadata={
                    "agent_version": "1.0",
                    "processing_type": "entity_validation",
                    "validation_criteria": {
                        "min_confidence": self.min_confidence,
                        "min_length": self.min_length,
                        "max_length": self.max_length
                    }
                }
            )
            
        except Exception as e:
            raise AgentError(f"Failed to validate entities: {str(e)}", self.name)
    
    async def _validate_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single entity against quality criteria."""
        entity_text = entity.get("text", "")
        entity_type = entity.get("type", "")
        confidence = entity.get("confidence", 0.0)
        
        validation_result = {
            "entity": entity,
            "is_valid": True,
            "issues": [],
            "quality_score": 1.0
        }
        
        # Check confidence threshold
        if confidence < self.min_confidence:
            validation_result["is_valid"] = False
            validation_result["issues"].append(f"Low confidence: {confidence} < {self.min_confidence}")
            validation_result["quality_score"] -= 0.3
        
        # Check length constraints
        if len(entity_text) < self.min_length:
            validation_result["is_valid"] = False
            validation_result["issues"].append(f"Too short: {len(entity_text)} < {self.min_length}")
            validation_result["quality_score"] -= 0.2
        
        if len(entity_text) > self.max_length:
            validation_result["is_valid"] = False
            validation_result["issues"].append(f"Too long: {len(entity_text)} > {self.max_length}")
            validation_result["quality_score"] -= 0.2
        
        # Check for blacklisted words
        if entity_text.lower() in self.blacklist_words:
            validation_result["is_valid"] = False
            validation_result["issues"].append(f"Blacklisted word: {entity_text}")
            validation_result["quality_score"] -= 0.5
        
        # Check for empty or whitespace-only text
        if not entity_text.strip():
            validation_result["is_valid"] = False
            validation_result["issues"].append("Empty or whitespace-only text")
            validation_result["quality_score"] = 0.0
        
        # Check for valid entity type
        try:
            EntityType(entity_type)
        except ValueError:
            validation_result["is_valid"] = False
            validation_result["issues"].append(f"Invalid entity type: {entity_type}")
            validation_result["quality_score"] -= 0.3
        
        # Ensure quality score doesn't go below 0
        validation_result["quality_score"] = max(0.0, validation_result["quality_score"])
        
        return validation_result
    
    def _generate_validation_summary(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of validation results."""
        total_entities = len(validation_results)
        valid_count = sum(1 for result in validation_results if result["is_valid"])
        invalid_count = total_entities - valid_count
        
        # Collect all issues
        all_issues = []
        for result in validation_results:
            all_issues.extend(result["issues"])
        
        # Count issue types
        issue_counts = {}
        for issue in all_issues:
            issue_type = issue.split(":")[0] if ":" in issue else issue
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        # Calculate average quality score
        quality_scores = [result["quality_score"] for result in validation_results]
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            "total_entities": total_entities,
            "valid_entities": valid_count,
            "invalid_entities": invalid_count,
            "validation_rate": valid_count / total_entities if total_entities > 0 else 0,
            "average_quality_score": avg_quality_score,
            "issue_distribution": issue_counts,
            "total_issues": len(all_issues)
        }
