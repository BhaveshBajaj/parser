"""Tests for the AgentRegistry and mock agents."""
import pytest
from unittest.mock import AsyncMock, patch

from app.models.document import Document, DocumentStatus
from app.services.agent_registry import (
    AgentRegistry, agent_registry,
    MockParser, MockSummarizer, MockValidator
)


@pytest.fixture
def test_document():
    """Create a test document."""
    return Document(
        id="test-doc-123",
        filename="test.txt",
        content_type="text/plain",
        size=100,
    )


@pytest.mark.asyncio
async def test_mock_parser(test_document):
    """Test the MockParser agent."""
    parser = MockParser()
    result = await parser.process(test_document)
    
    assert result["status"] == "success"
    assert "content" in result
    assert test_document.filename in result["content"]
    assert "metadata" in result
    assert "pages" in result["metadata"]


@pytest.mark.asyncio
async def test_mock_summarizer(test_document):
    """Test the MockSummarizer agent."""
    summarizer = MockSummarizer()
    result = await summarizer.process(test_document)
    
    assert result["status"] == "success"
    assert "summary" in result
    assert test_document.filename in result["summary"]
    assert "key_points" in result
    assert isinstance(result["key_points"], list)


@pytest.mark.asyncio
async def test_mock_validator(test_document):
    """Test the MockValidator agent."""
    validator = MockValidator()
    result = await validator.process(test_document)
    
    assert result["status"] == "success"
    assert "is_valid" in result
    assert isinstance(result["is_valid"], bool)
    assert "issues" in result
    assert isinstance(result["issues"], list)
    assert "score" in result
    assert 0 <= result["score"] <= 1


def test_agent_registry_initialization():
    """Test that the agent registry initializes with default agents."""
    registry = AgentRegistry()
    agents = registry.list_agents()
    
    # Check that all default agents are registered
    agent_names = {agent["name"] for agent in agents}
    assert "mock_parser" in agent_names
    assert "mock_summarizer" in agent_names
    assert "mock_validator" in agent_names


@pytest.mark.asyncio
async def test_agent_registry_call_agent(test_document):
    """Test calling an agent through the registry."""
    registry = AgentRegistry()
    
    # Test calling the mock parser
    result = await registry.call_agent("mock_parser", test_document)
    assert result["status"] == "success"
    
    # Test calling the mock summarizer
    result = await registry.call_agent("mock_summarizer", test_document)
    assert result["status"] == "success"
    
    # Test calling a non-existent agent
    with pytest.raises(ValueError, match="Agent not found: non_existent_agent"):
        await registry.call_agent("non_existent_agent", test_document)


def test_agent_registry_register_unregister():
    """Test registering and unregistering agents."""
    registry = AgentRegistry()
    
    # Create a mock agent
    class TestAgent:
        name = "test_agent"
        description = "A test agent"
        
        async def process(self, document, **kwargs):
            return {"status": "success", "result": "test result"}
    
    # Register the test agent
    registry.register(TestAgent())
    assert "test_agent" in [agent["name"] for agent in registry.list_agents()]
    
    # Unregister the test agent
    registry.unregister("test_agent")
    assert "test_agent" not in [agent["name"] for agent in registry.list_agents()]
    
    # Test unregistering a non-existent agent (should not raise)
    registry.unregister("non_existent_agent")
