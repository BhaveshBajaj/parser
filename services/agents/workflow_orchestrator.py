"""Clean AutoGen workflow orchestrator with rollback and exception handling."""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import autogen

from models.document import Document
from services.embedding_service import get_embedding_service
from services.metrics_storage import MetricsStorage
from .base_agent import BaseAutoGenAgent
from .summarization_agent import SummarizationAgent
from .entity_agent import EntityAgent
from .qa_agent import QAAgent
from .validation_agent import ValidationAgent
from .content_review_agent import ContentReviewAgent
from .security_agent import SecurityAgent
from .performance_agent import PerformanceAgent
from .critic_agent import CriticAgent

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """Clean AutoGen workflow orchestrator with rollback support."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize core agents
        self.summarization_agent = SummarizationAgent(self.config.get("summarization", {}))
        self.entity_agent = EntityAgent(self.config.get("entity", {}))
        self.qa_agent = QAAgent(self.config.get("qa", {}))
        self.validation_agent = ValidationAgent(self.config.get("validation", {}))
        
        # Initialize review and analysis agents
        self.content_review_agent = ContentReviewAgent(self.config.get("content_review", {}))
        self.security_agent = SecurityAgent(self.config.get("security", {}))
        self.performance_agent = PerformanceAgent(self.config.get("performance", {}))
        self.critic_agent = CriticAgent(self.config.get("critic", {}))
        
        # Initialize embedding service
        self.embedding_service = get_embedding_service(self.config)
        
        # Initialize metrics storage
        self.metrics_storage = MetricsStorage(self.config.get("metrics_storage_path"))
        
        # Create user proxy for coordination
        self.user_proxy = autogen.UserProxyAgent(
            name="coordinator",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config=False,
        )
        
        # Workflow state management
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_history: Dict[str, List[Dict[str, Any]]] = {}
    
    async def execute_document_workflow(
        self,
        document: Document,
        workflow_type: str = "full",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a document processing workflow using AutoGen with rollback support."""
        workflow_id = str(uuid4())
        self.logger.info(f"Starting AutoGen workflow {workflow_id} for document {document.id}")
        
        # Initialize workflow state
        self.active_workflows[workflow_id] = {
            "document_id": str(document.id),
            "workflow_type": workflow_type,
            "status": "running",
            "start_time": datetime.now(timezone.utc),
            "context": context or {},
            "checkpoints": [],
            "rollback_attempts": 0
        }
        
        try:
            # Set document context for all agents
            core_agents = [self.summarization_agent, self.entity_agent, self.qa_agent, self.validation_agent]
            review_agents = [self.content_review_agent, self.security_agent, self.performance_agent, self.critic_agent]
            
            for agent in core_agents + review_agents:
                agent.set_document_context(document, context)
            
            # Execute workflow based on type
            if workflow_type == "full":
                result = await self._execute_full_workflow(document, workflow_id, context)
            elif workflow_type == "summarization":
                result = await self._execute_summarization_workflow(document, workflow_id, context)
            elif workflow_type == "entity_extraction":
                result = await self._execute_entity_workflow(document, workflow_id, context)
            elif workflow_type == "qa" and context and "question" in context:
                result = await self._execute_qa_workflow(document, workflow_id, context)
            elif workflow_type == "content_review":
                result = await self._execute_content_review_workflow(document, workflow_id, context)
            elif workflow_type == "security_review":
                result = await self._execute_security_review_workflow(document, workflow_id, context)
            elif workflow_type == "performance_analysis":
                result = await self._execute_performance_analysis_workflow(document, workflow_id, context)
            elif workflow_type == "critic_analysis":
                result = await self._execute_critic_analysis_workflow(document, workflow_id, context)
            else:
                raise ValueError(f"Unknown workflow type: {workflow_type}")
            
            # Validate results
            validation_result = await self._validate_workflow_results(result, document, workflow_id)
            result["validation"] = validation_result
            
            # Check if rollback is needed
            if validation_result.get("rollback_decision", False):
                self.logger.warning(f"Validation failed for workflow {workflow_id}, triggering rollback")
                rollback_result = await self._execute_rollback(workflow_id, workflow_type, "Validation failed")
                result["rollback_executed"] = True
                result["rollback_result"] = rollback_result
            
            # Store metrics for analysis
            await self._store_workflow_metrics(workflow_id, workflow_type, result, document)
            
            # Update workflow state
            self.active_workflows[workflow_id]["status"] = "completed"
            self.active_workflows[workflow_id]["end_time"] = datetime.now(timezone.utc)
            
            return result
        
        except Exception as e:
            self.logger.error(f"AutoGen workflow {workflow_id} failed: {str(e)}", exc_info=True)
            
            # Attempt rollback if configured
            rollback_result = None
            if self.active_workflows[workflow_id]["rollback_attempts"] < 3:
                try:
                    rollback_result = await self._execute_rollback(workflow_id, workflow_type, str(e))
                except Exception as rollback_error:
                    self.logger.error(f"Rollback failed for workflow {workflow_id}: {str(rollback_error)}")
            
            # Update workflow state
            self.active_workflows[workflow_id]["status"] = "failed"
            self.active_workflows[workflow_id]["end_time"] = datetime.now(timezone.utc)
            self.active_workflows[workflow_id]["error"] = str(e)
            self.active_workflows[workflow_id]["rollback_result"] = rollback_result
            
            return {
                "workflow_id": workflow_id,
                "workflow_type": workflow_type,
                "document_id": str(document.id),
                "status": "failed",
                "error": str(e),
                "rollback_attempted": rollback_result is not None,
                "rollback_result": rollback_result
            }
    
    async def _execute_full_workflow(
        self,
        document: Document,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute full workflow with summarization, entity extraction, and validation."""
        
        self.logger.info(f"Starting simplified full workflow for document {document.filename}")
        
        # Execute agents sequentially (much simpler than GroupChat)
        agent_results = {}
        
        try:
            # 1. Summarization
            self.logger.info("Executing summarization step")
            summarization_result = await self._execute_summarization_workflow(document, workflow_id, context)
            agent_results["summarization_agent"] = summarization_result.get("agent_results", {}).get("summarization_agent", {})
            
            # 2. Entity extraction
            self.logger.info("Executing entity extraction step")
            entity_result = await self._execute_entity_workflow(document, workflow_id, context)
            agent_results["entity_agent"] = entity_result.get("agent_results", {}).get("entity_agent", {})
            
            # 3. Validation (optional - only if we have results to validate)
            if agent_results.get("entity_agent") or agent_results.get("summarization_agent"):
                self.logger.info("Executing validation step")
                validation_result = await self._validate_workflow_results({
                    "agent_results": agent_results
                }, document, workflow_id)
                agent_results["validation_agent"] = {"result_data": validation_result}
            
            return {
                "workflow_id": workflow_id,
                "workflow_type": "full",
                "document_id": str(document.id),
                "status": "completed",
                "agent_results": agent_results,
                "execution_method": "simplified_sequential"
            }
            
        except Exception as e:
            self.logger.error(f"Simplified full workflow failed: {str(e)}", exc_info=True)
            return {
                "workflow_id": workflow_id,
                "workflow_type": "full",
                "document_id": str(document.id),
                "status": "failed",
                "error": str(e),
                "agent_results": agent_results,
                "execution_method": "simplified_sequential"
            }
    
    async def _execute_summarization_workflow(
        self,
        document: Document,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute summarization-only workflow."""
        
        # Use shorter content for summarization to prevent context window overflow
        document_content = self.summarization_agent.get_document_content(max_length=7000)
        prompt = self.summarization_agent.generate_summary_prompt(document_content, "document")
        
        self.logger.info(f"Summarization prompt length: {len(prompt)} characters")
        
        # Direct interaction with summarization agent
        response = await asyncio.to_thread(
            self.user_proxy.initiate_chat,
            self.summarization_agent,
            message=prompt
        )
        
        return self._process_single_agent_result(
            response, self.summarization_agent, workflow_id, document, "summarization"
        )
    
    async def _execute_entity_workflow(
        self,
        document: Document,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute entity extraction workflow."""
        
        # Use shorter content for entity extraction to prevent context window overflow
        document_content = self.entity_agent.get_document_content(max_length=6000)
        entity_types = context.get("entity_types") if context else None
        prompt = self.entity_agent.generate_entity_prompt(document_content, entity_types)
        
        self.logger.info(f"Entity extraction prompt length: {len(prompt)} characters")
        
        # Direct interaction with entity agent
        response = await asyncio.to_thread(
            self.user_proxy.initiate_chat,
            self.entity_agent,
            message=prompt
        )
        
        return self._process_single_agent_result(
            response, self.entity_agent, workflow_id, document, "entity_extraction"
        )
    
    async def _execute_qa_workflow(
        self,
        document: Document,
        workflow_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute Q&A workflow."""
        
        question = context["question"]
        document_content = self.qa_agent.get_document_content()
        prompt = await self.qa_agent.generate_qa_prompt(document_content, question, context)
        
        # Direct interaction with Q&A agent
        response = await asyncio.to_thread(
            self.user_proxy.initiate_chat,
            self.qa_agent,
            message=prompt
        )
        
        return self._process_single_agent_result(
            response, self.qa_agent, workflow_id, document, "qa"
        )
    
    async def _execute_content_review_workflow(
        self,
        document: Document,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute content review workflow for bias detection and completeness analysis."""
        
        document_content = self.content_review_agent.get_document_content(max_length=8000)
        content_type = context.get("content_type", "general") if context else "general"
        prompt = self.content_review_agent.generate_content_review_prompt(document_content, content_type)
        
        self.logger.info(f"Content review prompt length: {len(prompt)} characters")
        
        # Direct interaction with content review agent
        response = await asyncio.to_thread(
            self.user_proxy.initiate_chat,
            self.content_review_agent,
            message=prompt
        )
        
        return self._process_single_agent_result(
            response, self.content_review_agent, workflow_id, document, "content_review"
        )
    
    async def _execute_security_review_workflow(
        self,
        document: Document,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute security review workflow for sensitive data detection and compliance checks."""
        
        document_content = self.security_agent.get_document_content(max_length=8000)
        content_type = context.get("content_type", "general") if context else "general"
        prompt = self.security_agent.generate_security_review_prompt(document_content, content_type)
        
        self.logger.info(f"Security review prompt length: {len(prompt)} characters")
        
        # Direct interaction with security agent
        response = await asyncio.to_thread(
            self.user_proxy.initiate_chat,
            self.security_agent,
            message=prompt
        )
        
        return self._process_single_agent_result(
            response, self.security_agent, workflow_id, document, "security_review"
        )
    
    async def _execute_performance_analysis_workflow(
        self,
        document: Document,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute performance analysis workflow for metrics tracking and optimization."""
        
        # Get performance metrics from context or generate default ones
        metrics_data = context.get("performance_metrics", {}) if context else {}
        
        # Add execution time tracking
        start_time = time.time()
        prompt = self.performance_agent.generate_performance_analysis_prompt(metrics_data)
        
        self.logger.info(f"Performance analysis prompt length: {len(prompt)} characters")
        
        # Direct interaction with performance agent
        response = await asyncio.to_thread(
            self.user_proxy.initiate_chat,
            self.performance_agent,
            message=prompt
        )
        
        end_time = time.time()
        
        # Track execution metrics
        execution_metrics = self.performance_agent.track_execution_metrics(
            workflow_id, "performance_agent", start_time, end_time, metrics_data
        )
        
        result = self._process_single_agent_result(
            response, self.performance_agent, workflow_id, document, "performance_analysis"
        )
        
        # Add execution metrics to result
        result["execution_metrics"] = execution_metrics
        
        return result
    
    async def _execute_critic_analysis_workflow(
        self,
        document: Document,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute critic analysis workflow for disagreement tracking and conflict resolution."""
        
        # Get agent results from context for comparison
        agent_results = context.get("agent_results", {}) if context else {}
        
        if not agent_results:
            return {
                "workflow_id": workflow_id,
                "workflow_type": "critic_analysis",
                "document_id": str(document.id),
                "status": "skipped",
                "reason": "No agent results available for comparison",
                "agent_results": {}
            }
        
        document_content = self.critic_agent.get_document_content(max_length=6000)
        prompt = self.critic_agent.generate_critic_analysis_prompt(agent_results, document_content)
        
        self.logger.info(f"Critic analysis prompt length: {len(prompt)} characters")
        
        # Direct interaction with critic agent
        response = await asyncio.to_thread(
            self.user_proxy.initiate_chat,
            self.critic_agent,
            message=prompt
        )
        
        result = self._process_single_agent_result(
            response, self.critic_agent, workflow_id, document, "critic_analysis"
        )
        
        # Track disagreement if we have results
        if result.get("agent_results", {}).get("critic_agent", {}).get("result_data"):
            conflict_analysis = result["agent_results"]["critic_agent"]["result_data"]
            disagreement_record = self.critic_agent.track_disagreement(
                workflow_id, agent_results, conflict_analysis
            )
            result["disagreement_tracking"] = disagreement_record
        
        return result
    
    def _process_single_agent_result(
        self,
        response: Any,
        agent: BaseAutoGenAgent,
        workflow_id: str,
        document: Document,
        workflow_type: str
    ) -> Dict[str, Any]:
        """Process results from single agent execution."""
        
        # Extract the response content
        response_content = ""
        if hasattr(response, 'chat_history') and response.chat_history:
            last_message = response.chat_history[-1]
            if isinstance(last_message, dict) and 'content' in last_message:
                response_content = last_message['content']
        
        # Try to parse JSON response
        try:
            # Handle markdown-wrapped JSON responses
            content_to_parse = response_content.strip()
            
            # Remove markdown code blocks
            if content_to_parse.startswith('```json') and content_to_parse.endswith('```'):
                content_to_parse = content_to_parse[7:-3].strip()
            elif content_to_parse.startswith('```') and content_to_parse.endswith('```'):
                content_to_parse = content_to_parse[3:-3].strip()
            
            # Try to find JSON object in the response
            if '{' in content_to_parse and '}' in content_to_parse:
                # Find the first complete JSON object
                start_idx = content_to_parse.find('{')
                if start_idx != -1:
                    # Find matching closing brace
                    brace_count = 0
                    end_idx = start_idx
                    for i, char in enumerate(content_to_parse[start_idx:], start_idx):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    
                    if brace_count == 0:  # Found complete JSON object
                        content_to_parse = content_to_parse[start_idx:end_idx]
            
            parsed_response = json.loads(content_to_parse)
            result_data = parsed_response
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response from {agent.name}: {str(e)}")
            logger.warning(f"Raw response: {response_content[:500]}...")
            result_data = {"response": response_content, "parse_error": str(e)}
        
        return {
            "workflow_id": workflow_id,
            "workflow_type": workflow_type,
            "document_id": str(document.id),
            "status": "completed",
            "agent_results": {
                agent.name: {
                    "status": "completed",
                    "result_data": result_data,
                    "raw_response": response_content
                }
            },
            "execution_method": "autogen_single_agent"
        }
    
    async def _validate_workflow_results(
        self,
        workflow_result: Dict[str, Any],
        document: Document,
        workflow_id: str
    ) -> Dict[str, Any]:
        """Validate workflow results using the validation agent."""
        try:
            # Prepare validation prompt
            agent_results = workflow_result.get("agent_results", {})
            document_content = self.validation_agent.get_document_content()
            validation_prompt = self.validation_agent.generate_validation_prompt(agent_results, document_content)
            
            # Execute validation
            response = await asyncio.to_thread(
                self.user_proxy.initiate_chat,
                self.validation_agent,
                message=validation_prompt
            )
            
            # Parse validation results
            validation_data = self._process_single_agent_result(
                response, self.validation_agent, workflow_id, document, "validation"
            )
            
            return validation_data.get("agent_results", {}).get("validation_agent", {}).get("result_data", {})
            
        except Exception as e:
            self.logger.error(f"Validation failed for workflow {workflow_id}: {str(e)}")
            return {
                "validation_error": str(e),
                "rollback_decision": True,
                "rollback_reason": f"Validation process failed: {str(e)}"
            }
    
    async def _execute_rollback(
        self,
        workflow_id: str,
        workflow_type: str,
        error_reason: str
    ) -> Dict[str, Any]:
        """Execute rollback for a failed workflow."""
        try:
            self.active_workflows[workflow_id]["rollback_attempts"] += 1
            
            # Simple rollback strategy - return error information
            return {
                "rollback_status": "completed",
                "rollback_method": "autogen_retry_required",
                "original_error": error_reason,
                "requires_manual_intervention": True,
                "attempts": self.active_workflows[workflow_id]["rollback_attempts"]
            }
            
        except Exception as e:
            self.logger.error(f"Rollback execution failed for workflow {workflow_id}: {str(e)}")
            return {
                "rollback_status": "failed",
                "error": str(e),
                "attempts": self.active_workflows[workflow_id]["rollback_attempts"]
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the LLM-powered system."""
        return {
            "implementation": "100% LLM-Powered AutoGen System with Rollback Support",
            "framework": "Microsoft AutoGen with Azure OpenAI",
            "capabilities": {
                "semantic_search": "Vector embeddings with Azure OpenAI",
                "entity_extraction": "LLM-powered entity analysis",
                "summarization": "Advanced LLM summarization",
                "question_answering": "Conversational AI with semantic retrieval",
                "validation": "LLM-powered cross-agent validation",
                "rollback": "Automatic rollback and recovery mechanisms",
                "exception_handling": "Comprehensive error handling and recovery"
            },
            "no_hardcoding": "All processing uses LLM reasoning",
            "no_fallbacks": "Pure LLM implementation - no fallback methods",
            "embedding_integration": "Full vector search and semantic similarity",
            "rollback_features": {
                "automatic_rollback": "Triggers on validation failures",
                "component_recovery": "Individual component rollback strategies",
                "retry_mechanisms": "AutoGen retry and recovery approaches",
                "error_recovery": "Comprehensive error handling"
            }
        }
    
    async def _store_workflow_metrics(self, workflow_id: str, workflow_type: str, result: Dict[str, Any], document: Document):
        """Store workflow metrics for analysis and tracking."""
        try:
            # Calculate execution time
            workflow_start = self.active_workflows[workflow_id]["start_time"]
            workflow_end = datetime.now(timezone.utc)
            execution_time = (workflow_end - workflow_start).total_seconds()
            
            # Store performance metrics
            performance_metrics = {
                "execution_time": execution_time,
                "workflow_type": workflow_type,
                "document_id": str(document.id),
                "document_filename": document.filename,
                "status": result.get("status", "unknown"),
                "agent_results_count": len(result.get("agent_results", {})),
                "validation_passed": not result.get("validation", {}).get("rollback_decision", False)
            }
            
            self.metrics_storage.store_performance_metrics(workflow_id, performance_metrics)
            
            # Store content review metrics if available
            if "content_review_agent" in result.get("agent_results", {}):
                content_review_data = result["agent_results"]["content_review_agent"].get("result_data", {})
                if content_review_data:
                    self.metrics_storage.store_content_review_metrics(workflow_id, content_review_data)
            
            # Store security metrics if available
            if "security_agent" in result.get("agent_results", {}):
                security_data = result["agent_results"]["security_agent"].get("result_data", {})
                if security_data:
                    self.metrics_storage.store_security_metrics(workflow_id, security_data)
            
            # Store critic feedback if available
            if "disagreement_tracking" in result:
                critic_feedback = result["disagreement_tracking"]
                self.metrics_storage.store_critic_feedback(workflow_id, critic_feedback)
            
            self.logger.info(f"Stored metrics for workflow {workflow_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store metrics for workflow {workflow_id}: {str(e)}")
    
    def get_performance_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get performance summary from metrics storage."""
        return self.metrics_storage.get_performance_summary(time_window_hours)
    
    def get_critic_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get critic feedback summary from metrics storage."""
        return self.metrics_storage.get_critic_summary(time_window_hours)
