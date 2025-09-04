"""Tests for the document service."""
import pytest
from uuid import UUID
import os
import shutil

from models.document import Document, DocumentStatus
from services.document_service import DocumentService, document_service


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    yield str(data_dir)
    # Cleanup
    shutil.rmtree(data_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_create_document(temp_data_dir):
    """Test creating a new document."""
    # Initialize service with temp directory
    service = DocumentService(data_dir=temp_data_dir)
    
    # Create a test document
    doc = Document(
        filename="test.txt",
        content_type="text/plain",
        size=100,
    )
    
    # Test creating the document
    file_content = b"This is a test file."
    created_doc = await service.create_document(doc, file_content)
    
    # Verify the document was created
    assert created_doc.id is not None
    assert created_doc.filename == "test.txt"
    assert created_doc.status == DocumentStatus.UPLOADED
    
    # Verify the file was saved
    doc_dir = os.path.join(temp_data_dir, str(created_doc.id))
    assert os.path.exists(doc_dir)
    assert os.path.exists(os.path.join(doc_dir, "test.txt"))
    
    # Verify the document is in the index
    retrieved_doc = await service.get_document(str(created_doc.id))
    assert retrieved_doc is not None
    assert retrieved_doc.id == created_doc.id


@pytest.mark.asyncio
async def test_update_document(temp_data_dir):
    """Test updating a document."""
    service = DocumentService(data_dir=temp_data_dir)
    
    # Create a test document
    doc = Document(
        filename="test.txt",
        content_type="text/plain",
        size=100,
    )
    file_content = b"Test content"
    created_doc = await service.create_document(doc, file_content)
    
    # Update the document
    update_data = {
        "status": DocumentStatus.PROCESSING,
        "summary": "Test summary",
    }
    
    updated_doc = await service.update_document(
        str(created_doc.id),
        DocumentUpdate(**update_data)
    )
    
    # Verify the update
    assert updated_doc.status == DocumentStatus.PROCESSING
    assert updated_doc.summary == "Test summary"
    assert updated_doc.updated_at > created_doc.updated_at
    
    # Verify the update was persisted
    retrieved_doc = await service.get_document(str(created_doc.id))
    assert retrieved_doc.status == DocumentStatus.PROCESSING
    assert retrieved_doc.summary == "Test summary"


@pytest.mark.asyncio
async def test_list_documents(temp_data_dir):
    """Test listing documents."""
    service = DocumentService(data_dir=temp_data_dir)
    
    # Create test documents
    doc1 = Document(
        filename="test1.txt",
        content_type="text/plain",
        size=100,
    )
    
    doc2 = Document(
        filename="test2.txt",
        content_type="text/plain",
        size=200,
    )
    
    await service.create_document(doc1, b"Content 1")
    await service.create_document(doc2, b"Content 2")
    
    # List documents
    documents = await service.list_documents()
    
    # Verify the documents were listed
    assert len(documents) == 2
    filenames = {doc.filename for doc in documents}
    assert "test1.txt" in filenames
    assert "test2.txt" in filenames
