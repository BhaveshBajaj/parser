"""Base agent class for document processing workflows."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from ..entity_extractor import Entity, EntityType
from ...models.document import Document, DocumentStatus

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Status of an agent execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class AgentError(Exception):
    """Custom exception for agent-related errors."""
    
    def __init__(self, message: str, agent_name: str, error_code: str = "AGENT_ERROR"):
        self.message = message
        self.agent_name = agent_name
        self.error_code = error_code
        super().__init__(f"[{agent_name}] {message}")


@dataclass
class AgentResult:
    """Result of an agent execution."""
    
    agent_name: str
    status: AgentStatus
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    rollback_data: Optional[Dict[str, Any]] = None
    
    def is_successful(self) -> bool:
        """Check if the agent execution was successful."""
        return self.status == AgentStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if the agent execution failed."""
        return self.status == AgentStatus.FAILED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "status": self.status.value,
            "result_data": self.result_data,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "rollback_data": self.rollback_data
        }


class BaseAgent(ABC):
    """Base class for all document processing agents."""
    
    def __init__(self, name: str, description: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base agent.
        
        Args:
            name: Unique name for the agent
            description: Human-readable description of the agent's purpose
            config: Configuration dictionary for the agent
        """
        self.name = name
        self.description = description
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    @abstractmethod
    async def process(
        self, 
        document: Document, 
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResult:
        """
        Process a document and return the result.
        
        Args:
            document: The document to process
            context: Additional context from previous agents
            **kwargs: Additional arguments
            
        Returns:
            AgentResult containing the processing results
        """
        pass
    
    @abstractmethod
    async def validate_input(self, document: Document, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate that the input is suitable for this agent.
        
        Args:
            document: The document to validate
            context: Additional context
            
        Returns:
            True if input is valid, False otherwise
        """
        pass
    
    async def rollback(self, document: Document, rollback_data: Dict[str, Any]) -> bool:
        """
        Rollback changes made by this agent.
        
        Args:
            document: The document to rollback
            rollback_data: Data needed for rollback
            
        Returns:
            True if rollback was successful, False otherwise
        """
        self.logger.warning(f"Rollback not implemented for agent: {self.name}")
        return False
    
    def _create_result(
        self,
        status: AgentStatus,
        result_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        error_code: Optional[str] = None,
        execution_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rollback_data: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """Create an AgentResult with the given parameters."""
        return AgentResult(
            agent_name=self.name,
            status=status,
            result_data=result_data or {},
            error_message=error_message,
            error_code=error_code,
            execution_time=execution_time,
            metadata=metadata or {},
            rollback_data=rollback_data
        )
    
    def _log_processing_start(self, document: Document, context: Optional[Dict[str, Any]] = None):
        """Log the start of processing."""
        self.logger.info(f"Starting processing for document: {document.id}")
        if context:
            self.logger.debug(f"Context: {context}")
    
    def _log_processing_end(self, result: AgentResult):
        """Log the end of processing."""
        if result.is_successful():
            self.logger.info(f"Successfully completed processing for agent: {self.name}")
        else:
            self.logger.error(f"Failed processing for agent: {self.name}, error: {result.error_message}")
    
    async def _execute_with_error_handling(
        self,
        document: Document,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResult:
        """Execute the agent with proper error handling and timing."""
        import time
        start_time = time.time()
        
        try:
            self._log_processing_start(document, context)
            
            # Validate input
            if not await self.validate_input(document, context):
                return self._create_result(
                    status=AgentStatus.FAILED,
                    error_message="Input validation failed",
                    error_code="VALIDATION_ERROR"
                )
            
            # Process the document
            result = await self.process(document, context, **kwargs)
            result.execution_time = time.time() - start_time
            
            self._log_processing_end(result)
            return result
            
        except AgentError as e:
            execution_time = time.time() - start_time
            result = self._create_result(
                status=AgentStatus.FAILED,
                error_message=e.message,
                error_code=e.error_code,
                execution_time=execution_time
            )
            self._log_processing_end(result)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = self._create_result(
                status=AgentStatus.FAILED,
                error_message=f"Unexpected error: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                execution_time=execution_time
            )
            self.logger.error(f"Unexpected error in agent {self.name}: {str(e)}", exc_info=True)
            self._log_processing_end(result)
            return result
