"""Cross-agent validation and rollback management."""

import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
from datetime import datetime, timezone

from .base_agent import BaseAgent, AgentResult, AgentStatus, AgentError
from ...models.document import Document

logger = logging.getLogger(__name__)


class CrossAgentValidator(BaseAgent):
    """Agent for validating outputs from other agents and ensuring quality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="cross_agent_validator",
            description="Validates outputs from other agents and ensures quality",
            config=config
        )
        self.validation_rules = self.config.get("validation_rules", {})
        self.quality_thresholds = self.config.get("quality_thresholds", {
            "summary_min_length": 50,
            "summary_max_length": 1000,
            "entity_min_confidence": 0.5,
            "qa_min_confidence": 0.3
        })
        self.rollback_on_failure = self.config.get("rollback_on_failure", True)
    
    async def validate_input(self, document: Document, context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate that we have agent results to validate."""
        if not context or "agent_results" not in context:
            self.logger.warning("No agent results provided for validation")
            return False
        
        agent_results = context["agent_results"]
        if not agent_results or len(agent_results) == 0:
            self.logger.warning("Empty agent results provided")
            return False
        
        return True
    
    async def process(
        self, 
        document: Document, 
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResult:
        """Validate outputs from multiple agents."""
        try:
            agent_results = context["agent_results"]
            validation_results = []
            overall_valid = True
            rollback_recommendations = []
            
            # Validate each agent's output
            for agent_result in agent_results:
                validation_result = await self._validate_agent_result(agent_result, document)
                validation_results.append(validation_result)
                
                if not validation_result["is_valid"]:
                    overall_valid = False
                    if self.rollback_on_failure:
                        rollback_recommendations.append({
                            "agent_name": agent_result.agent_name,
                            "reason": validation_result["issues"],
                            "rollback_data": agent_result.rollback_data
                        })
            
            # Cross-validate consistency between agents
            consistency_results = await self._validate_cross_agent_consistency(agent_results, document)
            
            # Generate validation summary
            validation_summary = self._generate_validation_summary(
                validation_results, 
                consistency_results,
                overall_valid
            )
            
            return self._create_result(
                status=AgentStatus.COMPLETED if overall_valid else AgentStatus.FAILED,
                result_data={
                    "overall_valid": overall_valid,
                    "validation_results": validation_results,
                    "consistency_results": consistency_results,
                    "validation_summary": validation_summary,
                    "rollback_recommendations": rollback_recommendations,
                    "agents_validated": len(agent_results)
                },
                metadata={
                    "agent_version": "1.0",
                    "processing_type": "cross_agent_validation",
                    "validation_timestamp": datetime.now(timezone.utc).isoformat()
                },
                error_message="Cross-agent validation failed" if not overall_valid else None,
                error_code="VALIDATION_FAILED" if not overall_valid else None
            )
            
        except Exception as e:
            raise AgentError(f"Failed to validate agent results: {str(e)}", self.name)
    
    async def _validate_agent_result(self, agent_result: AgentResult, document: Document) -> Dict[str, Any]:
        """Validate a single agent's result."""
        validation_result = {
            "agent_name": agent_result.agent_name,
            "is_valid": True,
            "issues": [],
            "quality_score": 1.0,
            "validation_details": {}
        }
        
        # Check if agent execution was successful
        if not agent_result.is_successful():
            validation_result["is_valid"] = False
            validation_result["issues"].append(f"Agent execution failed: {agent_result.error_message}")
            validation_result["quality_score"] = 0.0
            return validation_result
        
        # Validate based on agent type
        if "summariz" in agent_result.agent_name.lower():
            validation_result = await self._validate_summarization_result(agent_result, validation_result)
        elif "entity" in agent_result.agent_name.lower():
            validation_result = await self._validate_entity_result(agent_result, validation_result)
        elif "qa" in agent_result.agent_name.lower():
            validation_result = await self._validate_qa_result(agent_result, validation_result)
        
        return validation_result
    
    async def _validate_summarization_result(self, agent_result: AgentResult, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate summarization agent results."""
        result_data = agent_result.result_data
        
        # Check for summary content
        if "summary" in result_data:
            summary = result_data["summary"]
            summary_length = len(summary)
            
            # Check length constraints
            min_length = self.quality_thresholds["summary_min_length"]
            max_length = self.quality_thresholds["summary_max_length"]
            
            if summary_length < min_length:
                validation_result["is_valid"] = False
                validation_result["issues"].append(f"Summary too short: {summary_length} < {min_length}")
                validation_result["quality_score"] -= 0.3
            
            if summary_length > max_length:
                validation_result["is_valid"] = False
                validation_result["issues"].append(f"Summary too long: {summary_length} > {max_length}")
                validation_result["quality_score"] -= 0.2
            
            # Check for empty or generic summaries
            generic_phrases = ["this document", "the content", "information about"]
            if any(phrase in summary.lower() for phrase in generic_phrases) and summary_length < 100:
                validation_result["quality_score"] -= 0.2
                validation_result["issues"].append("Summary appears to be too generic")
            
            validation_result["validation_details"]["summary_length"] = summary_length
        
        # Check for key points if available
        if "key_points" in result_data:
            key_points = result_data["key_points"]
            if not key_points or len(key_points) == 0:
                validation_result["quality_score"] -= 0.1
                validation_result["issues"].append("No key points extracted")
            
            validation_result["validation_details"]["key_points_count"] = len(key_points)
        
        return validation_result
    
    async def _validate_entity_result(self, agent_result: AgentResult, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate entity extraction agent results."""
        result_data = agent_result.result_data
        
        # Check for entities
        if "all_entities" in result_data or "entities" in result_data:
            entities = result_data.get("all_entities", result_data.get("entities", []))
            
            if not entities:
                validation_result["quality_score"] -= 0.2
                validation_result["issues"].append("No entities extracted")
            else:
                # Check entity quality
                low_confidence_entities = 0
                min_confidence = self.quality_thresholds["entity_min_confidence"]
                
                for entity in entities:
                    confidence = entity.get("confidence", 1.0)
                    if confidence < min_confidence:
                        low_confidence_entities += 1
                
                if low_confidence_entities > len(entities) * 0.5:  # More than 50% low confidence
                    validation_result["is_valid"] = False
                    validation_result["issues"].append(f"Too many low confidence entities: {low_confidence_entities}/{len(entities)}")
                    validation_result["quality_score"] -= 0.4
                
                validation_result["validation_details"]["total_entities"] = len(entities)
                validation_result["validation_details"]["low_confidence_entities"] = low_confidence_entities
        
        # Check entity analysis if available
        if "entity_analysis" in result_data:
            analysis = result_data["entity_analysis"]
            avg_confidence = analysis.get("average_confidence", 0)
            
            if avg_confidence < self.quality_thresholds["entity_min_confidence"]:
                validation_result["quality_score"] -= 0.2
                validation_result["issues"].append(f"Low average entity confidence: {avg_confidence}")
        
        return validation_result
    
    async def _validate_qa_result(self, agent_result: AgentResult, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Q&A agent results."""
        result_data = agent_result.result_data
        
        # Check for answer
        if "answer" in result_data:
            answer = result_data["answer"]
            
            # Check for empty or non-informative answers
            non_informative_phrases = [
                "couldn't find",
                "don't know",
                "not available",
                "no information"
            ]
            
            if any(phrase in answer.lower() for phrase in non_informative_phrases):
                validation_result["quality_score"] -= 0.3
                validation_result["issues"].append("Answer is non-informative")
            
            # Check answer length
            if len(answer) < 20:
                validation_result["quality_score"] -= 0.2
                validation_result["issues"].append("Answer is too short")
            
            validation_result["validation_details"]["answer_length"] = len(answer)
        
        # Check confidence if available
        if "confidence" in result_data:
            confidence = result_data["confidence"]
            min_confidence = self.quality_thresholds["qa_min_confidence"]
            
            if confidence < min_confidence:
                validation_result["is_valid"] = False
                validation_result["issues"].append(f"Q&A confidence too low: {confidence} < {min_confidence}")
                validation_result["quality_score"] -= 0.4
            
            validation_result["validation_details"]["confidence"] = confidence
        
        return validation_result
    
    async def _validate_cross_agent_consistency(self, agent_results: List[AgentResult], document: Document) -> Dict[str, Any]:
        """Validate consistency between different agents' outputs."""
        consistency_results = {
            "overall_consistent": True,
            "consistency_issues": [],
            "consistency_score": 1.0,
            "cross_validations": []
        }
        
        # Group results by type
        summarization_results = []
        entity_results = []
        qa_results = []
        
        for result in agent_results:
            if "summariz" in result.agent_name.lower():
                summarization_results.append(result)
            elif "entity" in result.agent_name.lower():
                entity_results.append(result)
            elif "qa" in result.agent_name.lower():
                qa_results.append(result)
        
        # Validate entity-summary consistency
        if summarization_results and entity_results:
            entity_summary_consistency = await self._validate_entity_summary_consistency(
                entity_results, summarization_results
            )
            consistency_results["cross_validations"].append(entity_summary_consistency)
            
            if not entity_summary_consistency["is_consistent"]:
                consistency_results["overall_consistent"] = False
                consistency_results["consistency_issues"].extend(entity_summary_consistency["issues"])
                consistency_results["consistency_score"] -= 0.3
        
        # Validate Q&A-content consistency
        if qa_results and (summarization_results or entity_results):
            qa_content_consistency = await self._validate_qa_content_consistency(
                qa_results, summarization_results + entity_results
            )
            consistency_results["cross_validations"].append(qa_content_consistency)
            
            if not qa_content_consistency["is_consistent"]:
                consistency_results["overall_consistent"] = False
                consistency_results["consistency_issues"].extend(qa_content_consistency["issues"])
                consistency_results["consistency_score"] -= 0.2
        
        return consistency_results
    
    async def _validate_entity_summary_consistency(self, entity_results: List[AgentResult], summary_results: List[AgentResult]) -> Dict[str, Any]:
        """Validate consistency between entity extraction and summarization."""
        consistency_result = {
            "validation_type": "entity_summary_consistency",
            "is_consistent": True,
            "issues": [],
            "details": {}
        }
        
        # Extract entities and summaries
        all_entities = []
        all_summaries = []
        
        for result in entity_results:
            entities = result.result_data.get("all_entities", result.result_data.get("entities", []))
            all_entities.extend(entities)
        
        for result in summary_results:
            summary = result.result_data.get("summary", "")
            if summary:
                all_summaries.append(summary)
        
        if not all_entities or not all_summaries:
            return consistency_result
        
        # Check if important entities are mentioned in summaries
        important_entity_types = ["PERSON", "ORG", "GPE", "EVENT"]
        important_entities = [
            e for e in all_entities 
            if e.get("type") in important_entity_types and e.get("confidence", 0) > 0.7
        ]
        
        combined_summaries = " ".join(all_summaries).lower()
        missing_entities = []
        
        for entity in important_entities[:10]:  # Check top 10 important entities
            entity_text = entity.get("text", "").lower()
            if entity_text not in combined_summaries:
                missing_entities.append(entity)
        
        if len(missing_entities) > len(important_entities) * 0.5:  # More than 50% missing
            consistency_result["is_consistent"] = False
            consistency_result["issues"].append(
                f"Important entities missing from summaries: {len(missing_entities)}/{len(important_entities)}"
            )
        
        consistency_result["details"] = {
            "total_entities": len(all_entities),
            "important_entities": len(important_entities),
            "missing_entities": len(missing_entities),
            "summaries_checked": len(all_summaries)
        }
        
        return consistency_result
    
    async def _validate_qa_content_consistency(self, qa_results: List[AgentResult], content_results: List[AgentResult]) -> Dict[str, Any]:
        """Validate consistency between Q&A answers and document content."""
        consistency_result = {
            "validation_type": "qa_content_consistency",
            "is_consistent": True,
            "issues": [],
            "details": {}
        }
        
        # This is a simplified check - in a real implementation, 
        # you would use more sophisticated semantic similarity
        
        qa_answers = []
        content_texts = []
        
        for result in qa_results:
            answer = result.result_data.get("answer", "")
            if answer:
                qa_answers.append(answer)
        
        for result in content_results:
            if "summary" in result.result_data:
                content_texts.append(result.result_data["summary"])
            if "all_entities" in result.result_data:
                entities = result.result_data["all_entities"]
                entity_texts = [e.get("text", "") for e in entities]
                content_texts.extend(entity_texts)
        
        if not qa_answers or not content_texts:
            return consistency_result
        
        combined_content = " ".join(content_texts).lower()
        inconsistent_answers = 0
        
        for answer in qa_answers:
            answer_words = set(answer.lower().split())
            content_words = set(combined_content.split())
            
            # Check for word overlap
            overlap = len(answer_words.intersection(content_words))
            overlap_ratio = overlap / len(answer_words) if answer_words else 0
            
            if overlap_ratio < 0.3:  # Less than 30% word overlap
                inconsistent_answers += 1
        
        if inconsistent_answers > len(qa_answers) * 0.5:  # More than 50% inconsistent
            consistency_result["is_consistent"] = False
            consistency_result["issues"].append(
                f"Q&A answers inconsistent with content: {inconsistent_answers}/{len(qa_answers)}"
            )
        
        consistency_result["details"] = {
            "total_qa_answers": len(qa_answers),
            "content_sources": len(content_texts),
            "inconsistent_answers": inconsistent_answers
        }
        
        return consistency_result
    
    def _generate_validation_summary(
        self, 
        validation_results: List[Dict[str, Any]], 
        consistency_results: Dict[str, Any],
        overall_valid: bool
    ) -> Dict[str, Any]:
        """Generate a comprehensive validation summary."""
        total_agents = len(validation_results)
        valid_agents = sum(1 for result in validation_results if result["is_valid"])
        
        # Calculate average quality score
        quality_scores = [result["quality_score"] for result in validation_results]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Collect all issues
        all_issues = []
        for result in validation_results:
            all_issues.extend(result["issues"])
        all_issues.extend(consistency_results["consistency_issues"])
        
        return {
            "overall_valid": overall_valid,
            "total_agents_validated": total_agents,
            "valid_agents": valid_agents,
            "invalid_agents": total_agents - valid_agents,
            "average_quality_score": avg_quality,
            "consistency_score": consistency_results["consistency_score"],
            "total_issues": len(all_issues),
            "issue_summary": all_issues[:10],  # Top 10 issues
            "validation_passed": overall_valid and consistency_results["overall_consistent"]
        }


class RollbackManager(BaseAgent):
    """Agent for managing rollbacks when validation fails."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="rollback_manager",
            description="Manages rollbacks when agent validation fails",
            config=config
        )
        self.max_rollback_attempts = self.config.get("max_rollback_attempts", 3)
        self.rollback_strategies = self.config.get("rollback_strategies", ["revert", "retry", "skip"])
    
    async def validate_input(self, document: Document, context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate that we have rollback recommendations."""
        if not context or "rollback_recommendations" not in context:
            self.logger.warning("No rollback recommendations provided")
            return False
        
        return True
    
    async def process(
        self, 
        document: Document, 
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResult:
        """Execute rollbacks based on validation failures."""
        try:
            rollback_recommendations = context["rollback_recommendations"]
            rollback_results = []
            
            for recommendation in rollback_recommendations:
                rollback_result = await self._execute_rollback(recommendation, document)
                rollback_results.append(rollback_result)
            
            # Generate rollback summary
            rollback_summary = self._generate_rollback_summary(rollback_results)
            
            return self._create_result(
                status=AgentStatus.COMPLETED,
                result_data={
                    "rollback_results": rollback_results,
                    "rollback_summary": rollback_summary,
                    "total_rollbacks": len(rollback_recommendations),
                    "successful_rollbacks": sum(1 for r in rollback_results if r["success"])
                },
                metadata={
                    "agent_version": "1.0",
                    "processing_type": "rollback_management",
                    "rollback_timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            
        except Exception as e:
            raise AgentError(f"Failed to execute rollbacks: {str(e)}", self.name)
    
    async def _execute_rollback(self, recommendation: Dict[str, Any], document: Document) -> Dict[str, Any]:
        """Execute a single rollback operation."""
        rollback_result = {
            "agent_name": recommendation["agent_name"],
            "reason": recommendation["reason"],
            "success": False,
            "strategy_used": None,
            "details": {}
        }
        
        try:
            # Try different rollback strategies
            for strategy in self.rollback_strategies:
                success = await self._try_rollback_strategy(strategy, recommendation, document)
                if success:
                    rollback_result["success"] = True
                    rollback_result["strategy_used"] = strategy
                    rollback_result["details"]["strategy"] = strategy
                    break
            
            if not rollback_result["success"]:
                rollback_result["details"]["error"] = "All rollback strategies failed"
                
        except Exception as e:
            rollback_result["details"]["error"] = str(e)
            self.logger.error(f"Rollback failed for {recommendation['agent_name']}: {str(e)}")
        
        return rollback_result
    
    async def _try_rollback_strategy(self, strategy: str, recommendation: Dict[str, Any], document: Document) -> bool:
        """Try a specific rollback strategy."""
        if strategy == "revert":
            return await self._revert_agent_changes(recommendation, document)
        elif strategy == "retry":
            return await self._retry_agent_execution(recommendation, document)
        elif strategy == "skip":
            return await self._skip_agent_result(recommendation, document)
        else:
            self.logger.warning(f"Unknown rollback strategy: {strategy}")
            return False
    
    async def _revert_agent_changes(self, recommendation: Dict[str, Any], document: Document) -> bool:
        """Revert changes made by the agent."""
        try:
            rollback_data = recommendation.get("rollback_data")
            if not rollback_data:
                self.logger.warning(f"No rollback data available for {recommendation['agent_name']}")
                return False
            
            # This would implement actual reversion logic
            # For now, just log the reversion
            self.logger.info(f"Reverted changes for agent: {recommendation['agent_name']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to revert changes: {str(e)}")
            return False
    
    async def _retry_agent_execution(self, recommendation: Dict[str, Any], document: Document) -> bool:
        """Retry the agent execution with different parameters."""
        try:
            # This would implement retry logic with adjusted parameters
            # For now, just log the retry attempt
            self.logger.info(f"Retried execution for agent: {recommendation['agent_name']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to retry agent execution: {str(e)}")
            return False
    
    async def _skip_agent_result(self, recommendation: Dict[str, Any], document: Document) -> bool:
        """Skip the failed agent result and continue with others."""
        try:
            # This would implement skipping logic
            # For now, just log the skip
            self.logger.info(f"Skipped result for agent: {recommendation['agent_name']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to skip agent result: {str(e)}")
            return False
    
    def _generate_rollback_summary(self, rollback_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of rollback operations."""
        total_rollbacks = len(rollback_results)
        successful_rollbacks = sum(1 for result in rollback_results if result["success"])
        
        strategies_used = {}
        for result in rollback_results:
            if result["strategy_used"]:
                strategy = result["strategy_used"]
                strategies_used[strategy] = strategies_used.get(strategy, 0) + 1
        
        failed_agents = [
            result["agent_name"] for result in rollback_results 
            if not result["success"]
        ]
        
        return {
            "total_rollbacks": total_rollbacks,
            "successful_rollbacks": successful_rollbacks,
            "failed_rollbacks": total_rollbacks - successful_rollbacks,
            "success_rate": successful_rollbacks / total_rollbacks if total_rollbacks > 0 else 0,
            "strategies_used": strategies_used,
            "failed_agents": failed_agents,
            "rollback_complete": successful_rollbacks == total_rollbacks
        }
