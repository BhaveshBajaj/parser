"""Agent registry for managing document processing agents."""
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from uuid import UUID
import logging

from ..models.document import Document, DocumentStatus

logger = logging.getLogger(__name__)


@runtime_checkable
class Agent(Protocol):
    """Protocol for document processing agents."""
    
    name: str
    description: str
    
    async def process(self, document: Document, **kwargs) -> Dict[str, Any]:
        """Process a document and return the result."""
        ...


class MockParser:
    """Mock document parser agent."""
    
    name = "mock_parser"
    description = "Mock document parser that extracts text from documents"
    
    async def process(self, document: Document, **kwargs) -> Dict[str, Any]:
        """Mock parsing a document."""
        logger.info(f"Mock parsing document: {document.id}")
        return {
            "status": "success",
            "content": f"Mock content extracted from {document.filename}",
            "metadata": {
                "pages": 5,
                "language": "en",
            },
        }


class MockSummarizer:
    """Mock document summarizer agent."""
    
    name = "mock_summarizer"
    description = "Mock document summarizer that generates summaries"
    
    async def process(self, document: Document, **kwargs) -> Dict[str, Any]:
        """Mock summarizing a document."""
        logger.info(f"Mock summarizing document: {document.id}")
        return {
            "status": "success",
            "summary": f"This is a mock summary of {document.filename}. "
                      f"The document contains important information about document processing.",
            "key_points": [
                "Document processing is important",
                "This is a mock summary",
                "The document was processed successfully"
            ],
        }


class MockValidator:
    """Mock document validator agent."""
    
    name = "mock_validator"
    description = "Mock document validator that checks document quality"
    
    async def process(self, document: Document, **kwargs) -> Dict[str, Any]:
        """Mock validating a document."""
        logger.info(f"Mock validating document: {document.id}")
        return {
            "status": "success",
            "is_valid": True,
            "issues": [],
            "score": 0.95,
        }


class AgentRegistry:
    """Registry for document processing agents."""
    
    def __init__(self):
        """Initialize the agent registry with default agents."""
        self._agents: Dict[str, Agent] = {}
        self._register_default_agents()
    
    def _register_default_agents(self) -> None:
        """Register the default set of agents."""
        self.register(MockParser())
        self.register(MockSummarizer())
        self.register(MockValidator())
    
    def register(self, agent: Agent) -> None:
        """Register an agent."""
        self._agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")
    
    def unregister(self, agent_name: str) -> None:
        """Unregister an agent by name."""
        if agent_name in self._agents:
            del self._agents[agent_name]
            logger.info(f"Unregistered agent: {agent_name}")
    
    def get_agent(self, agent_name: str) -> Optional[Agent]:
        """Get an agent by name."""
        return self._agents.get(agent_name)
    
    def list_agents(self) -> List[Dict[str, str]]:
        """List all registered agents."""
        return [
            {"name": agent.name, "description": agent.description}
            for agent in self._agents.values()
        ]
    
    async def call_agent(
        self, agent_name: str, document: Document, **kwargs
    ) -> Dict[str, Any]:
        """Call an agent to process a document."""
        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent not found: {agent_name}")
        
        logger.info(f"Calling agent: {agent_name} for document: {document.id}")
        result = await agent.process(document, **kwargs)
        return result


# Create a singleton instance
agent_registry = AgentRegistry()
