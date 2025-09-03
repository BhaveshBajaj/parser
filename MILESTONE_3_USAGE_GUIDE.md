# Milestone 3: Agent Workflows - Usage Guide

## Overview

Milestone 3 implements a comprehensive agent-based workflow system for document processing, featuring multi-agent orchestration, cross-agent validation, and rollback mechanisms. The system uses AutoGen framework principles and provides REST API endpoints for various workflow operations.

## Architecture

### Agent Types

1. **Summarization Agents**
   - `SectionSummarizationAgent`: Summarizes individual document sections
   - `DocumentSummarizationAgent`: Creates comprehensive document summaries
   - `CorpusSummarizationAgent`: Generates corpus-level summaries across multiple documents

2. **Entity Extraction Agents**
   - `EntityExtractionAgent`: Extracts named entities from document content
   - `EntityTaggingAgent`: Tags and categorizes extracted entities
   - `EntityValidationAgent`: Validates entity quality and accuracy

3. **Q&A Agents**
   - `DocumentQAAgent`: Answers questions about individual documents
   - `CorpusQAAgent`: Answers questions across multiple documents
   - `ContextualQAAgent`: Provides enhanced contextual question answering

4. **Validation & Management Agents**
   - `CrossAgentValidator`: Validates outputs from multiple agents
   - `RollbackManager`: Manages rollbacks when validation fails

### Workflow Orchestrator

The `WorkflowOrchestrator` coordinates agent execution with:
- Dependency management
- Parallel execution capabilities
- Timeout handling
- Cross-agent validation
- Automatic rollback on failures

## API Endpoints

### Document Workflow Execution

Execute agent workflows on individual documents:

```bash
POST /documents/{document_id}/workflows
```

**Request Body:**
```json
{
  "workflow_type": "full",
  "context": {
    "custom_parameter": "value"
  }
}
```

**Available Workflow Types:**
- `full`: Complete processing with all agents
- `summarization`: Summarization agents only
- `entity_extraction`: Entity extraction and tagging
- `qa`: Question answering (requires question in context)

**Example Response:**
```json
{
  "workflow_id": "123e4567-e89b-12d3-a456-426614174000",
  "workflow_type": "full",
  "document_id": "doc-123",
  "status": "completed",
  "execution_results": {
    "entity_extractor": {
      "agent_name": "entity_extractor",
      "status": "completed",
      "result_data": {
        "all_entities": [...],
        "total_entities": 25
      }
    }
  },
  "workflow_summary": {
    "total_agents": 5,
    "successful_agents": 5,
    "success_rate": 1.0
  }
}
```

### Document Q&A

Ask questions about specific documents:

```bash
POST /documents/{document_id}/qa
```

**Request Body:**
```json
{
  "question": "What are the main topics discussed in this document?",
  "context": {
    "include_entities": true
  }
}
```

### Corpus Workflows

Execute workflows across multiple documents:

```bash
POST /corpus/workflows
```

**Request Body:**
```json
{
  "document_ids": ["doc-1", "doc-2", "doc-3"],
  "workflow_type": "corpus_analysis",
  "context": {}
}
```

### Corpus Q&A

Ask questions across multiple documents:

```bash
POST /corpus/qa
```

**Request Body:**
```json
{
  "document_ids": ["doc-1", "doc-2"],
  "workflow_type": "corpus_qa",
  "context": {}
}
```

**Question Request:**
```json
{
  "question": "What patterns emerge across these documents?",
  "context": {}
}
```

### Workflow Status

Check workflow execution status:

```bash
GET /workflows/{workflow_id}/status
```

## Usage Examples

### 1. Complete Document Processing

```python
import requests

# Upload a document first
with open("document.pdf", "rb") as f:
    upload_response = requests.post(
        "http://localhost:8000/documents",
        files={"file": f}
    )
document_id = upload_response.json()["id"]

# Execute full workflow
workflow_response = requests.post(
    f"http://localhost:8000/documents/{document_id}/workflows",
    json={
        "workflow_type": "full",
        "context": {}
    }
)

print(f"Workflow ID: {workflow_response.json()['workflow_id']}")
```

### 2. Entity Extraction Only

```python
entity_response = requests.post(
    f"http://localhost:8000/documents/{document_id}/workflows",
    json={
        "workflow_type": "entity_extraction",
        "context": {
            "entity_types": ["PERSON", "ORG", "GPE"],
            "confidence_threshold": 0.7
        }
    }
)
```

### 3. Document Q&A

```python
qa_response = requests.post(
    f"http://localhost:8000/documents/{document_id}/qa",
    json={
        "question": "Who are the key people mentioned in this document?",
        "context": {
            "include_evidence": true
        }
    }
)

answers = qa_response.json()["answers"]
```

### 4. Corpus Analysis

```python
# Get multiple document IDs
doc_ids = ["doc-1", "doc-2", "doc-3"]

corpus_response = requests.post(
    "http://localhost:8000/corpus/workflows",
    json={
        "document_ids": doc_ids,
        "workflow_type": "corpus_analysis",
        "context": {}
    }
)

corpus_summary = corpus_response.json()["corpus_results"]
```

## Configuration

### Agent Configuration

Configure individual agents through the workflow orchestrator:

```python
from doc_processor.app.services.agents import WorkflowOrchestrator

config = {
    "entity_extractor": {
        "confidence_threshold": 0.8,
        "entity_types": ["PERSON", "ORG", "DATE"]
    },
    "document_summarizer": {
        "summary_length": 500,
        "include_key_points": True,
        "max_key_points": 5
    }
}

orchestrator = WorkflowOrchestrator(config)
```

### Workflow Configuration

```python
workflow_config = {
    "max_parallel_agents": 3,
    "default_timeout": 300,  # 5 minutes
    "enable_validation": True,
    "enable_rollback": True
}

orchestrator = WorkflowOrchestrator(workflow_config)
```

## Error Handling

The system provides comprehensive error handling:

### Agent-Level Errors

- **AgentTimeoutError**: Agent execution timeout
- **AgentValidationError**: Input validation failure
- **AgentConfigurationError**: Configuration issues
- **AgentResourceError**: Resource access problems

### Workflow-Level Error Recovery

1. **Retry Strategy**: Automatic retry for timeout and resource errors
2. **Fallback Strategy**: Use default values when agents fail
3. **Rollback Strategy**: Revert changes when validation fails

### Example Error Response

```json
{
  "workflow_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed_with_errors",
  "execution_results": {
    "entity_extractor": {
      "status": "failed",
      "error_message": "Execution timed out after 300 seconds",
      "error_code": "TIMEOUT_ERROR"
    }
  },
  "workflow_summary": {
    "total_agents": 5,
    "successful_agents": 4,
    "failed_agents": 1,
    "errors": [
      {
        "agent": "entity_extractor",
        "error": "Execution timed out after 300 seconds",
        "error_code": "TIMEOUT_ERROR"
      }
    ]
  }
}
```

## Cross-Agent Validation

The system includes automatic validation between agents:

### Validation Types

1. **Entity-Summary Consistency**: Ensures important entities are mentioned in summaries
2. **Q&A-Content Consistency**: Validates that answers are consistent with document content
3. **Quality Thresholds**: Checks confidence scores and content quality

### Rollback Triggers

- Low confidence scores across multiple agents
- Inconsistent results between agents
- Failed quality checks
- Resource access failures

## Performance Considerations

### Parallel Execution

Agents can run in parallel when dependencies allow:
- Entity extraction and section summarization run simultaneously
- Independent validation steps execute in parallel

### Timeout Management

- Default timeout: 5 minutes per agent
- Configurable per agent type
- Automatic timeout detection and handling

### Resource Management

- Connection pooling for external APIs
- Memory management for large documents
- Graceful degradation on resource constraints

## Monitoring and Logging

### LangSmith Integration

The system integrates with LangSmith for:
- Agent execution tracking
- Performance monitoring
- Error analysis
- Workflow visualization

### Logging Levels

- **INFO**: Successful operations and workflow progress
- **WARNING**: Non-critical issues and fallback usage
- **ERROR**: Agent failures and system errors
- **DEBUG**: Detailed execution information

## Extending the System

### Adding New Agents

1. Inherit from `BaseAgent`
2. Implement required methods (`process`, `validate_input`)
3. Add error handling with decorators
4. Register with the orchestrator

```python
from doc_processor.app.services.agents import BaseAgent, with_error_handling

class CustomAgent(BaseAgent):
    def __init__(self, config=None):
        super().__init__("custom_agent", "Custom processing agent", config)
    
    @with_error_handling("custom_agent")
    async def process(self, document, context=None, **kwargs):
        # Implementation here
        pass
    
    async def validate_input(self, document, context=None):
        # Validation logic here
        return True
```

### Custom Workflows

Create custom workflow types by extending the orchestrator:

```python
def _create_custom_workflow_steps(self):
    return [
        WorkflowStep(agent=self.agents["custom_agent"], timeout=120),
        WorkflowStep(
            agent=self.agents["entity_extractor"], 
            dependencies=["custom_agent"]
        )
    ]
```

## Best Practices

1. **Always validate inputs** before processing
2. **Use appropriate timeouts** for different agent types
3. **Enable validation and rollback** for production workflows
4. **Monitor agent performance** and adjust configurations
5. **Handle errors gracefully** with fallback strategies
6. **Use parallel execution** when possible for performance
7. **Log comprehensively** for debugging and monitoring

## Troubleshooting

### Common Issues

1. **Agent Timeouts**: Increase timeout values or optimize agent logic
2. **Validation Failures**: Check input quality and agent configurations
3. **Resource Errors**: Verify API keys and network connectivity
4. **Memory Issues**: Process large documents in smaller chunks

### Debug Mode

Enable debug logging for detailed execution information:

```python
import logging
logging.getLogger("doc_processor.app.services.agents").setLevel(logging.DEBUG)
```

This comprehensive agent workflow system provides a robust foundation for multi-agent document processing with proper error handling, validation, and recovery mechanisms.
