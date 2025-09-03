"""Workflow orchestration system for coordinating multiple agents."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4
from datetime import datetime, timezone

from .base_agent import BaseAgent, AgentResult, AgentStatus, AgentError
from .summarization_agents import SectionSummarizationAgent, DocumentSummarizationAgent, CorpusSummarizationAgent
from .entity_agents import EntityExtractionAgent, EntityTaggingAgent, EntityValidationAgent
from .qa_agents import DocumentQAAgent, CorpusQAAgent, ContextualQAAgent
from .validation_agents import CrossAgentValidator, RollbackManager
from ...models.document import Document, DocumentUpdate

logger = logging.getLogger(__name__)


class WorkflowStep:
    """Represents a single step in a workflow."""
    
    def __init__(
        self,
        agent: BaseAgent,
        dependencies: Optional[List[str]] = None,
        required: bool = True,
        parallel: bool = False,
        retry_count: int = 0,
        timeout: Optional[float] = None
    ):
        self.agent = agent
        self.dependencies = dependencies or []
        self.required = required
        self.parallel = parallel
        self.retry_count = retry_count
        self.timeout = timeout
        self.result: Optional[AgentResult] = None
        self.execution_start: Optional[datetime] = None
        self.execution_end: Optional[datetime] = None


class WorkflowOrchestrator:
    """Orchestrates the execution of multiple agents in a defined workflow."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.max_parallel_agents = self.config.get("max_parallel_agents", 3)
        self.default_timeout = self.config.get("default_timeout", 300)  # 5 minutes
        self.enable_validation = self.config.get("enable_validation", True)
        self.enable_rollback = self.config.get("enable_rollback", True)
        
        # Initialize agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all available agents."""
        self.agents = {
            # Summarization agents
            "section_summarizer": SectionSummarizationAgent(self.config.get("section_summarizer", {})),
            "document_summarizer": DocumentSummarizationAgent(self.config.get("document_summarizer", {})),
            "corpus_summarizer": CorpusSummarizationAgent(self.config.get("corpus_summarizer", {})),
            
            # Entity agents
            "entity_extractor": EntityExtractionAgent(self.config.get("entity_extractor", {})),
            "entity_tagger": EntityTaggingAgent(self.config.get("entity_tagger", {})),
            "entity_validator": EntityValidationAgent(self.config.get("entity_validator", {})),
            
            # Q&A agents
            "document_qa": DocumentQAAgent(self.config.get("document_qa", {})),
            "corpus_qa": CorpusQAAgent(self.config.get("corpus_qa", {})),
            "contextual_qa": ContextualQAAgent(self.config.get("contextual_qa", {})),
            
            # Validation agents
            "cross_validator": CrossAgentValidator(self.config.get("cross_validator", {})),
            "rollback_manager": RollbackManager(self.config.get("rollback_manager", {}))
        }
    
    async def execute_document_processing_workflow(
        self,
        document: Document,
        workflow_type: str = "full",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a complete document processing workflow."""
        workflow_id = str(uuid4())
        self.logger.info(f"Starting document processing workflow {workflow_id} for document {document.id}")
        
        try:
            # Define workflow steps based on type
            workflow_steps = self._create_workflow_steps(workflow_type)
            
            # Execute the workflow
            execution_results = await self._execute_workflow(
                workflow_steps,
                document,
                context or {},
                workflow_id
            )
            
            # Generate workflow summary
            workflow_summary = self._generate_workflow_summary(execution_results, workflow_id)
            
            return {
                "workflow_id": workflow_id,
                "workflow_type": workflow_type,
                "document_id": str(document.id),
                "execution_results": execution_results,
                "workflow_summary": workflow_summary,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Workflow {workflow_id} failed: {str(e)}", exc_info=True)
            return {
                "workflow_id": workflow_id,
                "workflow_type": workflow_type,
                "document_id": str(document.id),
                "error": str(e),
                "status": "failed",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _create_workflow_steps(self, workflow_type: str) -> List[WorkflowStep]:
        """Create workflow steps based on the workflow type."""
        if workflow_type == "full":
            return self._create_full_workflow_steps()
        elif workflow_type == "summarization":
            return self._create_summarization_workflow_steps()
        elif workflow_type == "entity_extraction":
            return self._create_entity_workflow_steps()
        elif workflow_type == "qa":
            return self._create_qa_workflow_steps()
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
    
    def _create_full_workflow_steps(self) -> List[WorkflowStep]:
        """Create steps for a full document processing workflow."""
        steps = [
            # Phase 1: Basic extraction (parallel)
            WorkflowStep(
                agent=self.agents["entity_extractor"],
                parallel=True,
                timeout=self.default_timeout
            ),
            WorkflowStep(
                agent=self.agents["section_summarizer"],
                parallel=True,
                timeout=self.default_timeout
            ),
            
            # Phase 2: Enhanced processing (depends on Phase 1)
            WorkflowStep(
                agent=self.agents["entity_tagger"],
                dependencies=["entity_extractor"],
                timeout=self.default_timeout
            ),
            WorkflowStep(
                agent=self.agents["document_summarizer"],
                dependencies=["section_summarizer", "entity_extractor"],
                timeout=self.default_timeout
            ),
            
            # Phase 3: Validation (depends on Phase 2)
            WorkflowStep(
                agent=self.agents["entity_validator"],
                dependencies=["entity_tagger"],
                timeout=self.default_timeout
            ),
        ]
        
        # Add validation step if enabled
        if self.enable_validation:
            steps.append(
                WorkflowStep(
                    agent=self.agents["cross_validator"],
                    dependencies=["document_summarizer", "entity_validator"],
                    timeout=self.default_timeout
                )
            )
        
        return steps
    
    def _create_summarization_workflow_steps(self) -> List[WorkflowStep]:
        """Create steps for summarization-focused workflow."""
        return [
            WorkflowStep(
                agent=self.agents["section_summarizer"],
                timeout=self.default_timeout
            ),
            WorkflowStep(
                agent=self.agents["document_summarizer"],
                dependencies=["section_summarizer"],
                timeout=self.default_timeout
            )
        ]
    
    def _create_entity_workflow_steps(self) -> List[WorkflowStep]:
        """Create steps for entity extraction workflow."""
        steps = [
            WorkflowStep(
                agent=self.agents["entity_extractor"],
                timeout=self.default_timeout
            ),
            WorkflowStep(
                agent=self.agents["entity_tagger"],
                dependencies=["entity_extractor"],
                timeout=self.default_timeout
            ),
            WorkflowStep(
                agent=self.agents["entity_validator"],
                dependencies=["entity_tagger"],
                timeout=self.default_timeout
            )
        ]
        return steps
    
    def _create_qa_workflow_steps(self) -> List[WorkflowStep]:
        """Create steps for Q&A workflow."""
        return [
            WorkflowStep(
                agent=self.agents["document_qa"],
                timeout=self.default_timeout
            ),
            WorkflowStep(
                agent=self.agents["contextual_qa"],
                dependencies=["document_qa"],
                timeout=self.default_timeout
            )
        ]
    
    async def _execute_workflow(
        self,
        workflow_steps: List[WorkflowStep],
        document: Document,
        context: Dict[str, Any],
        workflow_id: str
    ) -> Dict[str, Any]:
        """Execute the workflow steps."""
        execution_results = {}
        completed_agents = set()
        context_data = dict(context)  # Copy initial context
        
        # Create execution phases
        phases = self._organize_steps_into_phases(workflow_steps)
        
        for phase_num, phase_steps in enumerate(phases):
            self.logger.info(f"Executing workflow phase {phase_num + 1} with {len(phase_steps)} steps")
            
            # Execute steps in this phase
            phase_results = await self._execute_phase(
                phase_steps,
                document,
                context_data,
                workflow_id,
                phase_num + 1
            )
            
            # Update execution results and context
            execution_results.update(phase_results)
            
            # Update context with results from this phase
            for agent_name, result in phase_results.items():
                completed_agents.add(agent_name)
                if result.is_successful():
                    # Add agent results to context for next phase
                    context_data[f"{agent_name}_result"] = result.result_data
                    
                    # Special handling for certain agent types
                    if agent_name == "entity_extractor":
                        context_data["entities"] = result.result_data.get("all_entities", [])
                    elif agent_name == "section_summarizer":
                        context_data["section_summaries"] = result.result_data.get("summarized_sections", [])
        
        # Execute validation if enabled and we have results
        if self.enable_validation and len(execution_results) > 1:
            validation_result = await self._execute_validation(
                execution_results,
                document,
                context_data,
                workflow_id
            )
            execution_results["validation"] = validation_result
        
        return execution_results
    
    def _organize_steps_into_phases(self, workflow_steps: List[WorkflowStep]) -> List[List[WorkflowStep]]:
        """Organize workflow steps into execution phases based on dependencies."""
        phases = []
        remaining_steps = workflow_steps.copy()
        completed_agents = set()
        
        while remaining_steps:
            current_phase = []
            
            # Find steps that can be executed (dependencies met or parallel)
            for step in remaining_steps[:]:
                can_execute = (
                    not step.dependencies or 
                    all(dep in completed_agents for dep in step.dependencies) or
                    step.parallel
                )
                
                if can_execute:
                    current_phase.append(step)
                    remaining_steps.remove(step)
            
            if not current_phase:
                # Deadlock - circular dependencies or missing dependencies
                self.logger.error("Workflow deadlock detected - circular or missing dependencies")
                break
            
            phases.append(current_phase)
            
            # Mark agents as completed for next phase
            for step in current_phase:
                completed_agents.add(step.agent.name)
        
        return phases
    
    async def _execute_phase(
        self,
        phase_steps: List[WorkflowStep],
        document: Document,
        context: Dict[str, Any],
        workflow_id: str,
        phase_num: int
    ) -> Dict[str, AgentResult]:
        """Execute all steps in a phase."""
        phase_results = {}
        
        # Determine if we can run steps in parallel
        parallel_steps = [step for step in phase_steps if step.parallel or len(phase_steps) == 1]
        sequential_steps = [step for step in phase_steps if not step.parallel and len(phase_steps) > 1]
        
        # Execute parallel steps
        if parallel_steps:
            parallel_tasks = []
            for step in parallel_steps:
                task = self._execute_step_with_timeout(step, document, context, workflow_id)
                parallel_tasks.append((step.agent.name, task))
            
            # Wait for parallel tasks to complete
            for agent_name, task in parallel_tasks:
                try:
                    result = await task
                    phase_results[agent_name] = result
                except Exception as e:
                    self.logger.error(f"Parallel step {agent_name} failed: {str(e)}")
                    phase_results[agent_name] = AgentResult(
                        agent_name=agent_name,
                        status=AgentStatus.FAILED,
                        error_message=str(e)
                    )
        
        # Execute sequential steps
        for step in sequential_steps:
            try:
                result = await self._execute_step_with_timeout(step, document, context, workflow_id)
                phase_results[step.agent.name] = result
                
                # Update context with this result for next sequential step
                if result.is_successful():
                    context[f"{step.agent.name}_result"] = result.result_data
                
            except Exception as e:
                self.logger.error(f"Sequential step {step.agent.name} failed: {str(e)}")
                phase_results[step.agent.name] = AgentResult(
                    agent_name=step.agent.name,
                    status=AgentStatus.FAILED,
                    error_message=str(e)
                )
        
        return phase_results
    
    async def _execute_step_with_timeout(
        self,
        step: WorkflowStep,
        document: Document,
        context: Dict[str, Any],
        workflow_id: str
    ) -> AgentResult:
        """Execute a single workflow step with timeout."""
        step.execution_start = datetime.now(timezone.utc)
        
        try:
            # Execute with timeout
            timeout = step.timeout or self.default_timeout
            result = await asyncio.wait_for(
                step.agent._execute_with_error_handling(document, context),
                timeout=timeout
            )
            
            step.result = result
            step.execution_end = datetime.now(timezone.utc)
            
            return result
            
        except asyncio.TimeoutError:
            step.execution_end = datetime.now(timezone.utc)
            self.logger.error(f"Agent {step.agent.name} timed out after {timeout} seconds")
            return AgentResult(
                agent_name=step.agent.name,
                status=AgentStatus.FAILED,
                error_message=f"Execution timed out after {timeout} seconds",
                error_code="TIMEOUT_ERROR"
            )
        except Exception as e:
            step.execution_end = datetime.now(timezone.utc)
            self.logger.error(f"Agent {step.agent.name} failed: {str(e)}")
            return AgentResult(
                agent_name=step.agent.name,
                status=AgentStatus.FAILED,
                error_message=str(e),
                error_code="EXECUTION_ERROR"
            )
    
    async def _execute_validation(
        self,
        execution_results: Dict[str, AgentResult],
        document: Document,
        context: Dict[str, Any],
        workflow_id: str
    ) -> AgentResult:
        """Execute cross-agent validation."""
        try:
            # Prepare validation context
            validation_context = {
                **context,
                "agent_results": list(execution_results.values())
            }
            
            # Execute validation
            validation_result = await self.agents["cross_validator"]._execute_with_error_handling(
                document,
                validation_context
            )
            
            # Execute rollback if validation failed and rollback is enabled
            if (not validation_result.is_successful() and 
                self.enable_rollback and 
                validation_result.result_data.get("rollback_recommendations")):
                
                rollback_context = {
                    "rollback_recommendations": validation_result.result_data["rollback_recommendations"]
                }
                
                rollback_result = await self.agents["rollback_manager"]._execute_with_error_handling(
                    document,
                    rollback_context
                )
                
                # Add rollback results to validation result
                validation_result.result_data["rollback_result"] = rollback_result.result_data
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return AgentResult(
                agent_name="cross_validator",
                status=AgentStatus.FAILED,
                error_message=f"Validation execution failed: {str(e)}",
                error_code="VALIDATION_EXECUTION_ERROR"
            )
    
    def _generate_workflow_summary(self, execution_results: Dict[str, AgentResult], workflow_id: str) -> Dict[str, Any]:
        """Generate a comprehensive workflow execution summary."""
        total_agents = len(execution_results)
        successful_agents = sum(1 for result in execution_results.values() if result.is_successful())
        failed_agents = total_agents - successful_agents
        
        # Calculate execution times
        execution_times = {}
        total_execution_time = 0
        
        for agent_name, result in execution_results.items():
            if result.execution_time:
                execution_times[agent_name] = result.execution_time
                total_execution_time += result.execution_time
        
        # Collect errors
        errors = []
        for agent_name, result in execution_results.items():
            if result.error_message:
                errors.append({
                    "agent": agent_name,
                    "error": result.error_message,
                    "error_code": result.error_code
                })
        
        # Check validation status
        validation_passed = True
        if "validation" in execution_results:
            validation_result = execution_results["validation"]
            validation_passed = validation_result.is_successful()
        
        return {
            "workflow_id": workflow_id,
            "total_agents": total_agents,
            "successful_agents": successful_agents,
            "failed_agents": failed_agents,
            "success_rate": successful_agents / total_agents if total_agents > 0 else 0,
            "total_execution_time": total_execution_time,
            "average_execution_time": total_execution_time / total_agents if total_agents > 0 else 0,
            "execution_times": execution_times,
            "validation_passed": validation_passed,
            "errors": errors,
            "status": "completed" if failed_agents == 0 else "completed_with_errors",
            "completion_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def execute_corpus_workflow(
        self,
        documents: List[Document],
        workflow_type: str = "corpus_analysis",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a workflow across multiple documents (corpus-level)."""
        workflow_id = str(uuid4())
        self.logger.info(f"Starting corpus workflow {workflow_id} for {len(documents)} documents")
        
        try:
            # Process each document individually first
            document_results = []
            for doc in documents:
                doc_result = await self.execute_document_processing_workflow(
                    doc,
                    "summarization",  # Use lighter workflow for corpus processing
                    context
                )
                document_results.append(doc_result)
            
            # Execute corpus-level agents
            corpus_context = {
                **(context or {}),
                "corpus_documents": documents,
                "document_results": document_results
            }
            
            # Use a representative document for corpus processing
            representative_doc = documents[0] if documents else None
            if not representative_doc:
                raise ValueError("No documents provided for corpus workflow")
            
            corpus_results = {}
            
            # Execute corpus summarization
            if workflow_type in ["corpus_analysis", "corpus_summarization"]:
                corpus_summary_result = await self.agents["corpus_summarizer"]._execute_with_error_handling(
                    representative_doc,
                    corpus_context
                )
                corpus_results["corpus_summarizer"] = corpus_summary_result
            
            # Execute corpus Q&A if requested
            if workflow_type in ["corpus_analysis", "corpus_qa"] and context and "question" in context:
                corpus_qa_result = await self.agents["corpus_qa"]._execute_with_error_handling(
                    representative_doc,
                    corpus_context
                )
                corpus_results["corpus_qa"] = corpus_qa_result
            
            return {
                "workflow_id": workflow_id,
                "workflow_type": workflow_type,
                "corpus_size": len(documents),
                "document_results": document_results,
                "corpus_results": corpus_results,
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Corpus workflow {workflow_id} failed: {str(e)}", exc_info=True)
            return {
                "workflow_id": workflow_id,
                "workflow_type": workflow_type,
                "corpus_size": len(documents),
                "error": str(e),
                "status": "failed",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
