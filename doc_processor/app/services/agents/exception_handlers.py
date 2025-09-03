"""Exception handling utilities for agent workflows."""

import logging
from typing import Any, Dict, Optional, Type
from functools import wraps
import asyncio

from .base_agent import AgentError, AgentResult, AgentStatus

logger = logging.getLogger(__name__)


class AgentTimeoutError(AgentError):
    """Exception raised when an agent times out."""
    
    def __init__(self, agent_name: str, timeout_seconds: float):
        super().__init__(
            message=f"Agent execution timed out after {timeout_seconds} seconds",
            agent_name=agent_name,
            error_code="TIMEOUT_ERROR"
        )
        self.timeout_seconds = timeout_seconds


class AgentValidationError(AgentError):
    """Exception raised when agent input validation fails."""
    
    def __init__(self, agent_name: str, validation_message: str):
        super().__init__(
            message=f"Input validation failed: {validation_message}",
            agent_name=agent_name,
            error_code="VALIDATION_ERROR"
        )


class AgentConfigurationError(AgentError):
    """Exception raised when agent configuration is invalid."""
    
    def __init__(self, agent_name: str, config_issue: str):
        super().__init__(
            message=f"Configuration error: {config_issue}",
            agent_name=agent_name,
            error_code="CONFIGURATION_ERROR"
        )


class AgentResourceError(AgentError):
    """Exception raised when agent cannot access required resources."""
    
    def __init__(self, agent_name: str, resource: str, details: str = ""):
        message = f"Resource access failed: {resource}"
        if details:
            message += f" - {details}"
        super().__init__(
            message=message,
            agent_name=agent_name,
            error_code="RESOURCE_ERROR"
        )


def with_error_handling(
    agent_name: str,
    default_result: Optional[Dict[str, Any]] = None,
    reraise_on: Optional[tuple] = None
):
    """
    Decorator to add comprehensive error handling to agent methods.
    
    Args:
        agent_name: Name of the agent for error reporting
        default_result: Default result to return on error
        reraise_on: Tuple of exception types to reraise instead of handling
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
                
            except AgentError:
                # AgentErrors are already properly formatted, just reraise
                raise
                
            except asyncio.TimeoutError as e:
                logger.error(f"Timeout in {agent_name}: {str(e)}")
                raise AgentTimeoutError(agent_name, kwargs.get('timeout', 300))
                
            except ValueError as e:
                logger.error(f"Value error in {agent_name}: {str(e)}")
                raise AgentValidationError(agent_name, str(e))
                
            except KeyError as e:
                logger.error(f"Key error in {agent_name}: {str(e)}")
                raise AgentValidationError(agent_name, f"Missing required key: {str(e)}")
                
            except Exception as e:
                # Check if we should reraise this exception
                if reraise_on and isinstance(e, reraise_on):
                    raise
                
                logger.error(f"Unexpected error in {agent_name}: {str(e)}", exc_info=True)
                raise AgentError(
                    message=f"Unexpected error: {str(e)}",
                    agent_name=agent_name,
                    error_code="UNEXPECTED_ERROR"
                )
        
        return wrapper
    return decorator


def safe_execute_agent(
    agent_func,
    agent_name: str,
    fallback_result: Optional[AgentResult] = None,
    timeout: Optional[float] = None
):
    """
    Safely execute an agent function with comprehensive error handling.
    
    Args:
        agent_func: The agent function to execute
        agent_name: Name of the agent for error reporting
        fallback_result: Result to return if execution fails
        timeout: Timeout in seconds
        
    Returns:
        AgentResult with success or error information
    """
    async def execute():
        try:
            if timeout:
                result = await asyncio.wait_for(agent_func(), timeout=timeout)
            else:
                result = await agent_func()
            
            return result
            
        except AgentError as e:
            logger.error(f"Agent error in {agent_name}: {e.message}")
            return AgentResult(
                agent_name=agent_name,
                status=AgentStatus.FAILED,
                error_message=e.message,
                error_code=e.error_code
            )
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout in {agent_name} after {timeout} seconds")
            return AgentResult(
                agent_name=agent_name,
                status=AgentStatus.FAILED,
                error_message=f"Execution timed out after {timeout} seconds",
                error_code="TIMEOUT_ERROR"
            )
            
        except Exception as e:
            logger.error(f"Unexpected error in {agent_name}: {str(e)}", exc_info=True)
            return fallback_result or AgentResult(
                agent_name=agent_name,
                status=AgentStatus.FAILED,
                error_message=f"Unexpected error: {str(e)}",
                error_code="UNEXPECTED_ERROR"
            )
    
    return execute()


class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""
    
    async def can_recover(self, error: AgentError) -> bool:
        """Check if this strategy can recover from the given error."""
        return False
    
    async def recover(self, error: AgentError, context: Dict[str, Any]) -> AgentResult:
        """Attempt to recover from the error."""
        raise NotImplementedError


class RetryStrategy(ErrorRecoveryStrategy):
    """Recovery strategy that retries the operation."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    async def can_recover(self, error: AgentError) -> bool:
        """Retry on timeout and resource errors."""
        return error.error_code in ["TIMEOUT_ERROR", "RESOURCE_ERROR"]
    
    async def recover(self, error: AgentError, context: Dict[str, Any]) -> AgentResult:
        """Implement retry logic."""
        # This would implement actual retry logic
        # For now, just return a failed result
        return AgentResult(
            agent_name=error.agent_name,
            status=AgentStatus.FAILED,
            error_message=f"Retry recovery not implemented: {error.message}",
            error_code="RECOVERY_NOT_IMPLEMENTED"
        )


class FallbackStrategy(ErrorRecoveryStrategy):
    """Recovery strategy that uses fallback values."""
    
    def __init__(self, fallback_results: Dict[str, Dict[str, Any]]):
        self.fallback_results = fallback_results
    
    async def can_recover(self, error: AgentError) -> bool:
        """Can recover if we have fallback results for this agent."""
        return error.agent_name in self.fallback_results
    
    async def recover(self, error: AgentError, context: Dict[str, Any]) -> AgentResult:
        """Return fallback result."""
        fallback_data = self.fallback_results.get(error.agent_name, {})
        
        return AgentResult(
            agent_name=error.agent_name,
            status=AgentStatus.COMPLETED,
            result_data=fallback_data,
            metadata={
                "recovery_used": "fallback",
                "original_error": error.message
            }
        )


class ErrorRecoveryManager:
    """Manages error recovery strategies for agent workflows."""
    
    def __init__(self):
        self.strategies = [
            RetryStrategy(max_retries=2),
            FallbackStrategy({
                "entity_extractor": {"all_entities": [], "total_entities": 0},
                "section_summarizer": {"summarized_sections": []},
                "document_summarizer": {"summary": "Summary generation failed", "key_points": []},
                "document_qa": {"answer": "Unable to answer question", "confidence": 0.0}
            })
        ]
    
    async def attempt_recovery(self, error: AgentError, context: Dict[str, Any]) -> Optional[AgentResult]:
        """
        Attempt to recover from an agent error using available strategies.
        
        Args:
            error: The agent error to recover from
            context: Context information for recovery
            
        Returns:
            AgentResult if recovery was successful, None otherwise
        """
        for strategy in self.strategies:
            try:
                if await strategy.can_recover(error):
                    logger.info(f"Attempting recovery for {error.agent_name} using {strategy.__class__.__name__}")
                    result = await strategy.recover(error, context)
                    if result.is_successful():
                        logger.info(f"Successfully recovered from error in {error.agent_name}")
                        return result
                    else:
                        logger.warning(f"Recovery attempt failed for {error.agent_name}")
                        
            except Exception as e:
                logger.error(f"Recovery strategy {strategy.__class__.__name__} failed: {str(e)}")
                continue
        
        logger.error(f"All recovery strategies failed for {error.agent_name}")
        return None


def validate_agent_input(
    document,
    context: Optional[Dict[str, Any]] = None,
    required_fields: Optional[list] = None,
    agent_name: str = "unknown"
) -> None:
    """
    Validate agent input parameters.
    
    Args:
        document: Document to validate
        context: Context to validate
        required_fields: List of required fields in context
        agent_name: Name of the agent for error reporting
        
    Raises:
        AgentValidationError: If validation fails
    """
    if not document:
        raise AgentValidationError(agent_name, "Document is required")
    
    if not hasattr(document, 'id'):
        raise AgentValidationError(agent_name, "Document must have an ID")
    
    if required_fields and context:
        missing_fields = []
        for field in required_fields:
            if field not in context:
                missing_fields.append(field)
        
        if missing_fields:
            raise AgentValidationError(
                agent_name,
                f"Missing required context fields: {', '.join(missing_fields)}"
            )
    elif required_fields and not context:
        raise AgentValidationError(
            agent_name,
            f"Context is required with fields: {', '.join(required_fields)}"
        )


def log_agent_performance(
    agent_name: str,
    execution_time: float,
    result: AgentResult,
    context_size: Optional[int] = None
) -> None:
    """
    Log agent performance metrics.
    
    Args:
        agent_name: Name of the agent
        execution_time: Time taken to execute
        result: Agent result
        context_size: Size of input context
    """
    status = "SUCCESS" if result.is_successful() else "FAILED"
    
    log_message = f"Agent {agent_name} - Status: {status}, Time: {execution_time:.2f}s"
    
    if context_size:
        log_message += f", Context Size: {context_size}"
    
    if result.error_message:
        log_message += f", Error: {result.error_message}"
    
    if result.is_successful():
        logger.info(log_message)
    else:
        logger.error(log_message)


# Global error recovery manager instance
error_recovery_manager = ErrorRecoveryManager()
