"""AutoGen-based agents for document processing workflows - Legacy compatibility module."""

# Import the new modular structure for backward compatibility
from .base_agent import BaseAutoGenAgent as AutoGenDocumentAgent
from .summarization_agent import SummarizationAgent as AutoGenSummarizationAgent
from .entity_agent import EntityAgent as AutoGenEntityAgent
from .qa_agent import QAAgent as AutoGenQAAgent
from .validation_agent import ValidationAgent as AutoGenValidationAgent
from .workflow_orchestrator import WorkflowOrchestrator as AutoGenWorkflowOrchestrator

# Maintain backward compatibility
__all__ = [
    "AutoGenDocumentAgent",
    "AutoGenSummarizationAgent",
    "AutoGenEntityAgent", 
    "AutoGenQAAgent",
    "AutoGenValidationAgent",
    "AutoGenWorkflowOrchestrator"
]
