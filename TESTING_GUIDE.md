# Testing Guide for Milestone 3: Agent Workflows

## Prerequisites

1. **Start the FastAPI server:**
```bash
cd /Users/bhavesh/Desktop/parser
source venv/bin/activate
uvicorn doc_processor.app.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Verify the server is running:**
```bash
curl http://localhost:8000/health
```

## Testing Methods

### 1. API Testing with cURL

#### Upload a Test Document First
```bash
curl -X POST "http://localhost:8000/documents" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/uploads/test_doc.txt" \
  -F "metadata={\"test\": true}"
```

Save the returned document ID for testing.

#### Test Full Workflow
```bash
# Replace {document_id} with actual ID from upload
curl -X POST "http://localhost:8000/documents/{document_id}/workflows" \
  -H "Content-Type: application/json" \
  -H "accept: application/json" \
  -d '{
    "workflow_type": "full",
    "context": {}
  }'
```

#### Test Summarization Only
```bash
curl -X POST "http://localhost:8000/documents/{document_id}/workflows" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_type": "summarization",
    "context": {}
  }'
```

#### Test Entity Extraction Only
```bash
curl -X POST "http://localhost:8000/documents/{document_id}/workflows" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_type": "entity_extraction",
    "context": {
      "entity_types": ["PERSON", "ORG", "GPE"],
      "confidence_threshold": 0.7
    }
  }'
```

#### Test Document Q&A
```bash
curl -X POST "http://localhost:8000/documents/{document_id}/qa" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main topics discussed in this document?",
    "context": {
      "include_evidence": true
    }
  }'
```

#### Test Corpus Workflows
```bash
# First get multiple document IDs, then:
curl -X POST "http://localhost:8000/corpus/workflows" \
  -H "Content-Type: application/json" \
  -d '{
    "document_ids": ["doc-id-1", "doc-id-2"],
    "workflow_type": "corpus_analysis",
    "context": {}
  }'
```

### 2. Python Testing Script

Create a test script to automate testing:

```python
# test_workflows.py
import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_workflow_system():
    """Comprehensive test of the workflow system."""
    
    # 1. Upload test documents
    print("1. Uploading test documents...")
    doc_ids = []
    
    test_files = [
        "data/uploads/test_doc.txt",
        "data/uploads/test_doc_2.txt"
    ]
    
    for file_path in test_files:
        if Path(file_path).exists():
            with open(file_path, 'rb') as f:
                response = requests.post(
                    f"{BASE_URL}/documents",
                    files={"file": f},
                    data={"metadata": json.dumps({"test": True})}
                )
                if response.status_code == 202:
                    doc_id = response.json()["id"]
                    doc_ids.append(doc_id)
                    print(f"  ✓ Uploaded: {file_path} -> {doc_id}")
                else:
                    print(f"  ✗ Failed to upload: {file_path}")
    
    if not doc_ids:
        print("No documents uploaded. Exiting.")
        return
    
    # 2. Test individual workflows
    print("\n2. Testing individual workflows...")
    
    workflows_to_test = [
        ("full", {}),
        ("summarization", {}),
        ("entity_extraction", {
            "entity_types": ["PERSON", "ORG", "GPE"],
            "confidence_threshold": 0.5
        })
    ]
    
    for workflow_type, context in workflows_to_test:
        print(f"\n  Testing {workflow_type} workflow...")
        response = requests.post(
            f"{BASE_URL}/documents/{doc_ids[0]}/workflows",
            json={
                "workflow_type": workflow_type,
                "context": context
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"    ✓ Workflow ID: {result['workflow_id']}")
            print(f"    ✓ Status: {result['status']}")
            if result.get('workflow_summary'):
                summary = result['workflow_summary']
                print(f"    ✓ Success Rate: {summary.get('success_rate', 0):.2%}")
        else:
            print(f"    ✗ Failed: {response.status_code} - {response.text}")
    
    # 3. Test Q&A
    print("\n3. Testing Q&A...")
    
    questions = [
        "What is this document about?",
        "Who are the key people mentioned?",
        "What are the main topics?"
    ]
    
    for question in questions:
        print(f"\n  Asking: {question}")
        response = requests.post(
            f"{BASE_URL}/documents/{doc_ids[0]}/qa",
            json={
                "question": question,
                "context": {"include_evidence": True}
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"    ✓ Workflow ID: {result['workflow_id']}")
            answers = result.get('answers', {})
            for agent, answer_data in answers.items():
                if 'answer' in answer_data:
                    print(f"    ✓ {agent}: {answer_data['answer'][:100]}...")
        else:
            print(f"    ✗ Failed: {response.status_code}")
    
    # 4. Test corpus workflows
    if len(doc_ids) > 1:
        print("\n4. Testing corpus workflows...")
        
        response = requests.post(
            f"{BASE_URL}/corpus/workflows",
            json={
                "document_ids": doc_ids,
                "workflow_type": "corpus_analysis",
                "context": {}
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"    ✓ Corpus Workflow ID: {result['workflow_id']}")
            print(f"    ✓ Corpus Size: {result['corpus_size']}")
            print(f"    ✓ Status: {result['status']}")
        else:
            print(f"    ✗ Failed: {response.status_code}")
    
    print("\n✅ Testing completed!")
    return doc_ids

if __name__ == "__main__":
    test_workflow_system()
```

Run the test:
```bash
python test_workflows.py
```

### 3. Interactive Testing with FastAPI Docs

1. **Open the interactive API docs:**
   - Go to: http://localhost:8000/docs
   - This provides a web interface to test all endpoints

2. **Test workflow endpoints:**
   - Find the "Agent Workflow Endpoints" section
   - Use the "Try it out" buttons to test each endpoint
   - View request/response examples

### 4. Testing with Postman

Import this Postman collection:

```json
{
  "info": {
    "name": "Agent Workflows API",
    "description": "Test collection for Milestone 3 agent workflows"
  },
  "item": [
    {
      "name": "Upload Document",
      "request": {
        "method": "POST",
        "header": [],
        "body": {
          "mode": "formdata",
          "formdata": [
            {
              "key": "file",
              "type": "file",
              "src": "test_doc.txt"
            }
          ]
        },
        "url": {
          "raw": "{{base_url}}/documents",
          "host": ["{{base_url}}"],
          "path": ["documents"]
        }
      }
    },
    {
      "name": "Execute Full Workflow",
      "request": {
        "method": "POST",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          }
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"workflow_type\": \"full\",\n  \"context\": {}\n}"
        },
        "url": {
          "raw": "{{base_url}}/documents/{{document_id}}/workflows",
          "host": ["{{base_url}}"],
          "path": ["documents", "{{document_id}}", "workflows"]
        }
      }
    }
  ],
  "variable": [
    {
      "key": "base_url",
      "value": "http://localhost:8000"
    },
    {
      "key": "document_id",
      "value": "your-document-id-here"
    }
  ]
}
```

### 5. Unit Testing

Create unit tests for individual agents:

```python
# test_agents.py
import pytest
import asyncio
from doc_processor.app.services.agents import (
    SectionSummarizationAgent,
    EntityExtractionAgent,
    DocumentQAAgent,
    WorkflowOrchestrator
)
from doc_processor.app.models.document import Document, DocumentStatus
from uuid import uuid4

@pytest.fixture
def sample_document():
    return Document(
        id=uuid4(),
        filename="test.txt",
        content_type="text/plain",
        size=1000,
        status=DocumentStatus.PROCESSED,
        extra_data={
            "sections": [
                {
                    "title": "Introduction",
                    "content": "This is a test document about artificial intelligence and machine learning.",
                    "section_type": "section"
                },
                {
                    "title": "Main Content",
                    "content": "John Smith works at OpenAI in San Francisco. The company was founded in 2015.",
                    "section_type": "section"
                }
            ]
        }
    )

@pytest.mark.asyncio
async def test_section_summarization_agent(sample_document):
    """Test section summarization agent."""
    agent = SectionSummarizationAgent()
    
    result = await agent._execute_with_error_handling(sample_document)
    
    assert result.is_successful()
    assert "summarized_sections" in result.result_data
    assert len(result.result_data["summarized_sections"]) == 2

@pytest.mark.asyncio
async def test_entity_extraction_agent(sample_document):
    """Test entity extraction agent."""
    agent = EntityExtractionAgent()
    
    result = await agent._execute_with_error_handling(sample_document)
    
    assert result.is_successful()
    assert "all_entities" in result.result_data

@pytest.mark.asyncio
async def test_document_qa_agent(sample_document):
    """Test document Q&A agent."""
    agent = DocumentQAAgent()
    
    context = {"question": "Who works at OpenAI?"}
    result = await agent._execute_with_error_handling(sample_document, context)
    
    assert result.is_successful()
    assert "answer" in result.result_data
    assert "question" in result.result_data

@pytest.mark.asyncio
async def test_workflow_orchestrator(sample_document):
    """Test workflow orchestrator."""
    orchestrator = WorkflowOrchestrator()
    
    result = await orchestrator.execute_document_processing_workflow(
        document=sample_document,
        workflow_type="summarization"
    )
    
    assert "workflow_id" in result
    assert result["workflow_type"] == "summarization"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

Run the tests:
```bash
pytest test_agents.py -v
```

### 6. Load Testing

Test system performance with multiple concurrent requests:

```python
# load_test.py
import asyncio
import aiohttp
import time
import json

async def test_workflow_load(session, document_id, workflow_type="summarization"):
    """Test a single workflow request."""
    url = f"http://localhost:8000/documents/{document_id}/workflows"
    data = {
        "workflow_type": workflow_type,
        "context": {}
    }
    
    start_time = time.time()
    try:
        async with session.post(url, json=data) as response:
            result = await response.json()
            end_time = time.time()
            return {
                "status": response.status,
                "duration": end_time - start_time,
                "workflow_id": result.get("workflow_id"),
                "success": response.status == 200
            }
    except Exception as e:
        return {
            "status": 500,
            "duration": time.time() - start_time,
            "error": str(e),
            "success": False
        }

async def run_load_test(document_id, num_requests=10):
    """Run load test with multiple concurrent requests."""
    print(f"Running load test with {num_requests} concurrent requests...")
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            test_workflow_load(session, document_id)
            for _ in range(num_requests)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Analyze results
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        avg_duration = sum(r["duration"] for r in results) / len(results)
        
        print(f"\n📊 Load Test Results:")
        print(f"  Total Requests: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Success Rate: {successful/len(results):.2%}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Average Duration: {avg_duration:.2f}s")
        print(f"  Requests/Second: {len(results)/total_time:.2f}")

if __name__ == "__main__":
    # Replace with actual document ID
    document_id = "your-document-id-here"
    asyncio.run(run_load_test(document_id, 5))
```

## Expected Test Results

### Successful Workflow Response
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
        "total_entities": 15
      }
    },
    "section_summarizer": {
      "agent_name": "section_summarizer", 
      "status": "completed",
      "result_data": {
        "summarized_sections": [...],
        "total_sections": 3
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

### Successful Q&A Response
```json
{
  "document_id": "doc-123",
  "question": "What are the main topics?",
  "answers": {
    "document_qa": {
      "answer": "The main topics include...",
      "confidence": 0.85,
      "evidence": [...]
    }
  },
  "workflow_id": "qa-workflow-456",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Troubleshooting Common Issues

### 1. Agent Timeouts
```bash
# Check logs for timeout errors
tail -f fastapi_detailed.log | grep -i timeout
```

### 2. Validation Failures
```bash
# Check validation results in workflow response
curl -X POST ".../workflows" | jq '.execution_results.validation'
```

### 3. Missing Dependencies
```bash
# Install missing dependencies
pip install pyautogen>=0.2.0
```

### 4. Document Processing Issues
```bash
# Check document status first
curl http://localhost:8000/documents/{document_id}/status
```

## Monitoring and Debugging

### Check Server Logs
```bash
# Real-time log monitoring
tail -f fastapi_detailed.log

# Filter for specific agent
tail -f fastapi_detailed.log | grep "entity_extractor"

# Check for errors
tail -f fastapi_detailed.log | grep -i error
```

### Health Check Endpoints
```bash
# Basic health check
curl http://localhost:8000/health

# API root with endpoints
curl http://localhost:8000/
```

This comprehensive testing guide covers everything from basic API testing to load testing and debugging. Start with the simple cURL commands, then move to the Python scripts for more comprehensive testing.
