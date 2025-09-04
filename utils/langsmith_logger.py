"""LangSmith logging utilities."""
import os
from typing import Any, Dict, Optional, Union
from uuid import UUID, uuid4
import logging

from ..core.config import settings

logger = logging.getLogger(__name__)


try:
    from langsmith import Client
    from langsmith.schemas import Run, Example
    
    LANGCHAIN_TRACING_V2 = True
    LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY = settings.LANGSMITH_API_KEY
    LANGCHAIN_PROJECT = settings.LANGSMITH_PROJECT
    
    client = Client()
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class LangSmithLogger:
    """LangSmith logger for tracing and logging operations."""
    
    def __init__(self, enabled: bool = True):
        """Initialize the LangSmith logger."""
        self.enabled = enabled and LANGCHAIN_AVAILABLE and bool(settings.LANGSMITH_API_KEY)
        
        if self.enabled:
            logger.info("LangSmith logging is enabled")
        else:
            logger.info("LangSmith logging is disabled")
    
    def log_run(
        self,
        name: str,
        inputs: Dict[str, Any],
        outputs: Optional[Dict[str, Any]] = None,
        run_type: str = "chain",
        **kwargs
    ) -> Optional[UUID]:
        """Log a run to LangSmith."""
        if not self.enabled:
            return None
            
        try:
            run = Run(
                name=name,
                inputs=inputs,
                outputs=outputs or {},
                run_type=run_type,
                **kwargs
            )
            
            result = client.create_run(
                name=name,
                run=run,
                project_name=LANGCHAIN_PROJECT,
            )
            
            logger.debug(f"Logged run to LangSmith: {result.id}")
            return result.id
            
        except Exception as e:
            logger.error(f"Error logging to LangSmith: {str(e)}", exc_info=True)
            return None
    
    def log_chain(
        self,
        name: str,
        inputs: Dict[str, Any],
        outputs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[UUID]:
        """Log a chain run to LangSmith."""
        return self.log_run(name, inputs, outputs, run_type="chain", **kwargs)
    
    def log_llm(
        self,
        name: str,
        inputs: Dict[str, Any],
        outputs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[UUID]:
        """Log an LLM run to LangSmith."""
        return self.log_run(name, inputs, outputs, run_type="llm", **kwargs)
    
    def log_tool(
        self,
        name: str,
        inputs: Dict[str, Any],
        outputs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[UUID]:
        """Log a tool run to LangSmith."""
        return self.log_run(name, inputs, outputs, run_type="tool", **kwargs)


# Create a singleton instance
langsmith_logger = LangSmithLogger()
