"""Tests for the FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
import os
import json
from pathlib import Path

from app.main import app
from app.models.document import Document, DocumentStatus
from app.services.document_service import document_service

# Test client
client = TestClient(app)

# Test data
TEST_FILE_CONTENT = b"This is a test file."
TEST_FILENAME = "test.txt"
TEST_CONTENT_TYPE = "text/plain"


@pytest.fixture(autouse=True)
def setup_teardown():
    """Setup and teardown for each test."""
    # Setup: Clear the document service before each test
    if hasattr(document_service, 'documents'):
        document_service.documents.clear()
    
    # Create data directory if it doesn't exist
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    yield  # Run the test
    
    # Teardown: Clean up after each test
    if hasattr(document_service, 'documents'):
        document_service.documents.clear()
    
    # Clean up test files
    for item in data_dir.glob("*"):
        if item.is_dir():
            for file_item in item.glob("*"):
                file_item.unlink()
            item.rmdir()
    
    # Clean up documents.json if it exists
    docs_file = data_dir / "documents.json"
    if docs_file.exists():
        docs_file.unlink()


def test_health_check():
    ""Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_document():
    ""Test uploading a document."""
    # Prepare test file
    files = {
        "file": (TEST_FILENAME, TEST_FILE_CONTENT, TEST_CONTENT_TYPE)
    }
    
    # Send request
    response = client.post("/api/v1/upload", files=files)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["filename"] == TEST_FILENAME
    assert data["status"] == "uploaded"
    
    # Verify the document was saved
    doc_id = data["id"]
    doc = document_service.documents.get(doc_id)
    assert doc is not None
    assert doc.filename == TEST_FILENAME
    assert doc.status == DocumentStatus.UPLOADED


def test_get_document_status():
    ""Test getting document status."""
    # First, upload a document
    files = {"file": (TEST_FILENAME, TEST_FILE_CONTENT, TEST_CONTENT_TYPE)}
    upload_response = client.post("/api/v1/upload", files=files)
    doc_id = upload_response.json()["id"]
    
    # Now get the status
    response = client.get(f"/api/v1/status/{doc_id}")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == doc_id
    assert data["filename"] == TEST_FILENAME
    assert data["status"] == "uploaded"


def test_get_nonexistent_document_status():
    ""Test getting status for a non-existent document."""
    response = client.get("/api/v1/status/nonexistent-id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_get_document_summary_not_ready():
    ""Test getting a summary when the document isn't summarized yet."""
    # First, upload a document
    files = {"file": (TEST_FILENAME, TEST_FILE_CONTENT, TEST_CONTENT_TYPE)}
    upload_response = client.post("/api/v1/upload", files=files)
    doc_id = upload_response.json()["id"]
    
    # Try to get the summary (should fail)
    response = client.get(f"/api/v1/summaries/{doc_id}")
    
    # Check response
    assert response.status_code == 404
    assert response.json()["detail"] == "Summary not available"


def test_mock_qa():
    ""Test the mock QA endpoint."""
    # Test without document ID
    response = client.post(
        "/api/v1/qa",
        json={"question": "What is this document about?"}
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert "sources" in data
    assert isinstance(data["sources"], list)
    
    # Test with document ID
    response = client.post(
        "/api/v1/qa",
        json={
            "question": "What is this document about?",
            "doc_id": "test-doc-123"
        }
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) > 0  # Should have sources when doc_id is provided


def test_upload_invalid_file():
    ""Test uploading an invalid file."""
    # Test with no file
    response = client.post("/api/v1/upload")
    assert response.status_code == 422  # Validation error
    
    # Test with empty file
    files = {"file": ("", b"", "application/octet-stream")}
    response = client.post("/api/v1/upload", files=files)
    assert response.status_code == 400  # Bad request


def test_file_size_limit():
    ""Test the file size limit."""
    # Create a file larger than the limit (16MB)
    large_file_content = b"x" * (17 * 1024 * 1024)  # 17MB
    
    files = {
        "file": ("large_file.txt", large_file_content, "text/plain")
    }
    
    # This should fail due to file size limit
    response = client.post("/api/v1/upload", files=files)
    assert response.status_code == 413  # Payload Too Large
