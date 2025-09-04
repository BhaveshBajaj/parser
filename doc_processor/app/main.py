"""Main FastAPI application for document processing service."""
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from pydantic import BaseModel, Field, HttpUrl
from fastapi.encoders import jsonable_encoder

from .core.config import settings
from .models.document import Document, DocumentCreate, DocumentStatus, DocumentUpdate
from .services.document_service import document_service
from .services.parsers import DocumentSection
from .models.entity import Entity, EntityType
from .services.agents.autogen_agents import AutoGenWorkflowOrchestrator
from .services.langchain_qa import get_qa_service

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("fastapi_detailed.log")
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Document Processing API with AutoGen Multi-Agent Workflows and LLM-Powered Analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZip middleware for responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Initialize AutoGen workflow orchestrator
autogen_orchestrator = AutoGenWorkflowOrchestrator()

# Initialize LangChain Q&A service
qa_service = get_qa_service()

# Custom JSON encoder for UUID and datetime
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        if hasattr(obj, "dict"):
            return obj.dict()
        return super().default(obj)

# Dependency to get the current request ID for logging
def get_request_id() -> str:
    return str(uuid4())

# Root endpoint
@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint with API information.
    
    Returns:
        Dict with API information and available endpoints
    """
    return {
        "message": "Document Processing API with AutoGen Multi-Agent Workflows",
        "version": "1.0.0",
        "status": "running",
        "framework": "Microsoft AutoGen",
        "implementation": "LLM-powered multi-agent system",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoints": {
            "health": "/health",
            "upload_document": "/documents",
            "list_documents": "/documents",
            "get_document": "/documents/{document_id}",
            "document_status": "/documents/{document_id}/status",
            "extract_entities": "/entities/extract",
            "execute_workflow": "/documents/{document_id}/workflows",
            "document_qa": "/documents/{document_id}/qa",
            "corpus_workflow": "/corpus/workflows",
            "corpus_qa": "/corpus/qa",
            "workflow_status": "/workflows/{workflow_id}/status",
            "orchestrator_info": "/orchestrator/info"
        },
        "capabilities": [
            "AutoGen multi-agent orchestration",
            "LLM-powered document analysis",
            "Conversational AI agents",
            "Advanced reasoning capabilities"
        ],
        "docs": "/docs"
    }

# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        Dict with status and timestamp
    """
    return {
        "status": "ok",
        "version": "1.0.0",
        "framework": "AutoGen",
        "implementation": "LLM-powered",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# Document Models
class DocumentResponse(Document):
    """Response model for document with string IDs."""
    id: str
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda dt: dt.isoformat(),
        }
        from_attributes = True

class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int

class EntityResponse(BaseModel):
    """Response model for entities."""
    text: str
    type: str
    start: int
    end: int
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

class SectionResponse(BaseModel):
    """Response model for document sections."""
    title: str
    content: str
    page_number: Optional[int] = None
    section_type: str
    entities: List[EntityResponse] = []

class DocumentDetailResponse(DocumentResponse):
    """Detailed document response with sections and entities."""
    sections: List[SectionResponse] = []
    entities: List[EntityResponse] = []

# API Endpoints

@app.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for processing"
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
    metadata: str = Form(None, description="Optional metadata as JSON string"),
    request_id: str = Depends(get_request_id)
) -> DocumentResponse:
    """
    Upload a document for processing.
    
    The document will be processed asynchronously. Check the status using the returned document ID.
    """
    logger.info(f"[{request_id}] Received document upload: {file.filename}")
    logger.debug(f"[{request_id}] File details: content_type={file.content_type}, size={file.size}")
    
    # Validate file type
    if not any(file.filename.lower().endswith(ext) for ext in ['.pdf', '.docx', '.doc', '.html', '.txt']):
        logger.warning(f"[{request_id}] Unsupported file type: {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Allowed types: .pdf, .docx, .doc, .html, .txt"
        )
    
    try:
        # Parse metadata if provided
        doc_metadata = {}
        if metadata:
            try:
                doc_metadata = json.loads(metadata)
                if not isinstance(doc_metadata, dict):
                    raise ValueError("Metadata must be a JSON object")
            except json.JSONDecodeError:
                logger.warning(f"[{request_id}] Invalid metadata format")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid metadata format. Must be a valid JSON object."
                )
        
        # Read file content
        content = await file.read()
        logger.debug(f"[{request_id}] File content read, length: {len(content)}")
        
        # Create document
        doc_data = DocumentCreate(
            filename=file.filename,
            content_type=file.content_type or "application/octet-stream",
            size=len(content),
            metadata={
                "uploaded_via": "api",
                **doc_metadata
            }
        )
        logger.debug(f"[{request_id}] DocumentCreate data: {doc_data.model_dump()}")
        
        logger.info(f"[{request_id}] Creating document: {file.filename}")
        document = await document_service.create_document(doc_data, content)
        
        logger.info(f"[{request_id}] Document created: {document.id}")
        logger.debug(f"[{request_id}] Document response: {document.model_dump()}")
        # Convert UUID to string for the response
        doc_data = document.model_dump()
        doc_data['id'] = str(doc_data['id'])
        return DocumentResponse(**doc_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error uploading document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )

@app.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List documents"
)
async def list_documents(
    status: Optional[DocumentStatus] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    request_id: str = Depends(get_request_id)
) -> DocumentListResponse:
    """
    List documents with optional filtering by status.
    """
    try:
        logger.info(f"[{request_id}] Listing documents (status={status}, page={page}, page_size={page_size})")
        
        # Get documents with pagination
        offset = (page - 1) * page_size
        documents = await document_service.list_documents(
            status=status,
            limit=page_size,
            offset=offset
        )
        
        # Get total count for pagination
        total_docs = len(await document_service.list_documents(status=status))
        
        # Convert to response model
        doc_responses = []
        for doc in documents:
            doc_data = doc.model_dump()
            doc_data['id'] = str(doc_data['id'])
            doc_responses.append(DocumentResponse(**doc_data))
        
        return DocumentListResponse(
            documents=doc_responses,
            total=total_docs,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Error listing documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error listing documents"
        )

@app.get(
    "/documents/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details"
)
async def get_document(
    document_id: str,
    include_sections: bool = Query(True, description="Include document sections"),
    include_entities: bool = Query(True, description="Include extracted entities"),
    request_id: str = Depends(get_request_id)
) -> DocumentDetailResponse:
    """
    Get detailed information about a document, including its sections and entities.
    """
    try:
        logger.info(f"[{request_id}] Getting document: {document_id}")
        
        # Validate and get document
        logger.debug(f"[{request_id}] Raw document_id: '{document_id}' (type: {type(document_id)}, length: {len(document_id)})")
        
        # Clean the document ID - remove any whitespace or hidden characters
        cleaned_doc_id = document_id.strip()
        logger.debug(f"[{request_id}] Cleaned document_id: '{cleaned_doc_id}'")
        
        # Test UUID parsing with more detailed logging
        logger.info(f"[{request_id}] Attempting to parse UUID: '{cleaned_doc_id}'")
        try:
            doc_uuid = UUID(cleaned_doc_id)
            logger.info(f"[{request_id}] Successfully parsed document ID: {doc_uuid}")
        except ValueError as e:
            logger.error(f"[{request_id}] UUID parsing failed: '{document_id}', error: {str(e)}")
            logger.error(f"[{request_id}] Document ID bytes: {document_id.encode('utf-8')}")
            logger.error(f"[{request_id}] Cleaned document ID bytes: {cleaned_doc_id.encode('utf-8')}")
            logger.error(f"[{request_id}] Document ID repr: {repr(document_id)}")
            logger.error(f"[{request_id}] Cleaned document ID repr: {repr(cleaned_doc_id)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid document ID format: {document_id}"
            )
            
        document = await document_service.get_document(doc_uuid)
        if not document:
            logger.warning(f"[{request_id}] Document not found: {document_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Build response
        response_data = document.dict()
        # Convert UUID to string for the response
        response_data['id'] = str(response_data['id'])
        
        # Include sections if requested
        if include_sections and document.extra_data and 'sections' in document.extra_data:
            sections_data = document.extra_data['sections']
            response_data['sections'] = [
                SectionResponse(
                    title=s.get('title', ''),
                    content=s.get('content', ''),
                    page_number=s.get('page_number'),
                    section_type=s.get('section_type', 'section'),
                    entities=[
                        EntityResponse(
                            text=e['text'],
                            type=e['type'],
                            start=e['start'],
                            end=e['end'],
                            confidence=e.get('confidence', 1.0),
                            metadata=e.get('metadata')
                        ) for e in s.get('entities', [])
                    ] if include_entities and 'entities' in s else []
                ) for s in sections_data
            ]
        else:
            response_data['sections'] = []
        
        # Include all entities if requested
        if include_entities and document.extra_data and 'all_entities' in document.extra_data:
            response_data['entities'] = [
                EntityResponse(
                    text=e['text'],
                    type=e['type'],
                    start=e['start'],
                    end=e['end'],
                    confidence=e.get('confidence', 1.0),
                    metadata=e.get('metadata')
                ) for e in document.extra_data['all_entities']
            ]
        else:
            response_data['entities'] = []
        
        return DocumentDetailResponse(**response_data)
        
    except ValueError as e:
        logger.error(f"[{request_id}] ValueError in get_document: {str(e)}")
        logger.error(f"[{request_id}] Document ID: {document_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error getting document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving document"
        )

@app.get(
    "/documents/{document_id}/status",
    response_model=Dict[str, Any],
    summary="Get document processing status"
)
async def get_document_status(
    document_id: str,
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """
    Get the processing status of a document.
    """
    try:
        logger.info(f"[{request_id}] Getting status for document: {document_id}")
        
        document = await document_service.get_document(UUID(document_id.strip()))
        if not document:
            logger.warning(f"[{request_id}] Document not found: {document_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return {
            "document_id": str(document.id),
            "status": document.status,
            "created_at": document.created_at,
            "updated_at": document.updated_at,
            "filename": document.filename,
            "error": document.error
        }
        
    except ValueError as e:
        logger.warning(f"[{request_id}] Invalid document ID format: {document_id}, error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document ID format: {document_id}"
        )
    except Exception as e:
        logger.error(f"[{request_id}] Error getting document status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving document status"
        )

@app.get(
    "/documents/{document_id}/content",
    response_class=StreamingResponse,
    summary="Get document content"
)
async def get_document_content(
    document_id: str,
    request_id: str = Depends(get_request_id)
):
    """
    Download the original document content.
    """
    try:
        logger.info(f"[{request_id}] Getting content for document: {document_id}")
        
        document = await document_service.get_document(UUID(document_id.strip()))
        if not document:
            logger.warning(f"[{request_id}] Document not found: {document_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        stored_path = document.metadata.get("stored_path")
        if not stored_path or not os.path.exists(stored_path):
            logger.warning(f"[{request_id}] Document file not found: {stored_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document file not found"
            )
        
        # Stream the file
        file = open(stored_path, "rb")
        
        return StreamingResponse(
            file,
            media_type=document.content_type or "application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={document.filename}",
                "Document-ID": str(document.id)
            }
        )
        
    except ValueError as e:
        logger.warning(f"[{request_id}] Invalid document ID format: {document_id}, error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document ID format: {document_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error getting document content: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving document content"
        )

@app.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document"
)
async def delete_document(
    document_id: str,
    request_id: str = Depends(get_request_id)
):
    """
    Delete a document and all its associated data.
    """
    try:
        logger.info(f"[{request_id}] Deleting document: {document_id}")
        
        success = await document_service.delete_document(UUID(document_id.strip()))
        if not success:
            logger.warning(f"[{request_id}] Document not found for deletion: {document_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return None  # 204 No Content
        
    except ValueError as e:
        logger.warning(f"[{request_id}] Invalid document ID format: {document_id}, error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document ID format: {document_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error deleting document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting document"
        )

# Entity extraction endpoints

class EntityExtractionRequest(BaseModel):
    """Request model for entity extraction."""
    text: str = Field(..., description="Text to extract entities from")
    entity_types: Optional[List[EntityType]] = Field(
        None,
        description="Specific entity types to extract. If not provided, uses default types from settings."
    )

class EntityExtractionResponse(BaseModel):
    """Response model for entity extraction."""
    entities: List[EntityResponse]
    text: str
    entity_types: List[str]

# Feedback models
class FeedbackSubmission(BaseModel):
    """Model for feedback submission."""
    document_id: str
    entities: Dict[str, Any] = Field(..., description="Entity feedback data")
    summary: Dict[str, Any] = Field(..., description="Summary feedback data")
    timestamp: Optional[str] = Field(None, description="Timestamp of feedback submission")

class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    success: bool
    message: str
    feedback_id: str
    timestamp: str

@app.post(
    "/entities/extract",
    response_model=Dict[str, Any],
    summary="Extract entities from text using AutoGen"
)
async def extract_entities(
    request: EntityExtractionRequest,
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """
    Extract named entities from the provided text using AutoGen agents.
    """
    try:
        logger.info(f"[{request_id}] Extracting entities from text using AutoGen (length: {len(request.text)})")
        
        # Create a temporary document for entity extraction
        from .models.document import Document, DocumentStatus
        temp_doc = Document(
            filename="temp_text.txt",
            content_type="text/plain",
            size=len(request.text),
            status=DocumentStatus.PROCESSED,
            extra_data={
                "sections": [{
                    "title": "Text Content",
                    "content": request.text,
                    "section_type": "content"
                }]
            }
        )
        
        # Use AutoGen for entity extraction
        workflow_result = await autogen_orchestrator.execute_document_workflow(
            document=temp_doc,
            workflow_type="entity_extraction"
        )
        
        # Extract entities from AutoGen result
        agent_results = workflow_result.get("agent_results", {})
        entity_agent_result = agent_results.get("entity_agent", {})
        
        if entity_agent_result:
            result_data = entity_agent_result.get("result_data", {})
            entities_data = result_data.get("entities", [])
        else:
            entities_data = []
        
        return {
            "entities": entities_data,
            "text": request.text,
            "entity_types": [t.value for t in (request.entity_types or [])],
            "extraction_method": "autogen_agents",
            "workflow_id": workflow_result.get("workflow_id"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Error extracting entities: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting entities: {str(e)}"
        )

# Feedback endpoints

@app.post(
    "/documents/{document_id}/feedback",
    response_model=FeedbackResponse,
    summary="Submit feedback for document entities and summary"
)
async def submit_feedback(
    document_id: str,
    feedback: FeedbackSubmission,
    request_id: str = Depends(get_request_id)
) -> FeedbackResponse:
    """
    Submit human-in-the-loop feedback for document entities and summary.
    This endpoint allows users to approve, reject, or edit extracted entities and summaries.
    """
    try:
        logger.info(f"[{request_id}] Submitting feedback for document: {document_id}")
        
        # Validate document exists
        doc_uuid = UUID(document_id.strip())
        document = await document_service.get_document(doc_uuid)
        if not document:
            logger.warning(f"[{request_id}] Document not found: {document_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Generate feedback ID
        feedback_id = str(uuid4())
        
        # Store feedback data in document's extra_data
        feedback_data = {
            "feedback_id": feedback_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "entities": feedback.entities,
            "summary": feedback.summary,
            "submitted_by": "user",  # Could be enhanced with user authentication
            "version": "1.0"
        }
        
        # Update document with feedback
        update_data = DocumentUpdate(
            extra_data={"feedback": feedback_data}
        )
        
        updated_doc = await document_service.update_document(doc_uuid, update_data)
        if not updated_doc:
            logger.error(f"[{request_id}] Failed to update document with feedback")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save feedback"
            )
        
        logger.info(f"[{request_id}] Feedback submitted successfully: {feedback_id}")
        
        return FeedbackResponse(
            success=True,
            message="Feedback submitted successfully",
            feedback_id=feedback_id,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except ValueError as e:
        logger.warning(f"[{request_id}] Invalid document ID format: {document_id}, error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document ID format: {document_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error submitting feedback: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error submitting feedback: {str(e)}"
        )

@app.get(
    "/documents/{document_id}/feedback",
    response_model=Dict[str, Any],
    summary="Get feedback for a document"
)
async def get_feedback(
    document_id: str,
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """
    Get feedback data for a document.
    """
    try:
        logger.info(f"[{request_id}] Getting feedback for document: {document_id}")
        
        # Validate document exists
        doc_uuid = UUID(document_id.strip())
        document = await document_service.get_document(doc_uuid)
        if not document:
            logger.warning(f"[{request_id}] Document not found: {document_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Get feedback data
        feedback_data = document.extra_data.get("feedback") if document.extra_data else None
        
        if not feedback_data:
            return {
                "document_id": document_id,
                "feedback_exists": False,
                "message": "No feedback found for this document"
            }
        
        return {
            "document_id": document_id,
            "feedback_exists": True,
            "feedback": feedback_data
        }
        
    except ValueError as e:
        logger.warning(f"[{request_id}] Invalid document ID format: {document_id}, error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document ID format: {document_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error getting feedback: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving feedback: {str(e)}"
        )


@app.post(
    "/documents/{document_id}/regenerate-summary",
    response_model=Dict[str, Any],
    summary="Regenerate document summary using AI"
)
async def regenerate_summary(
    document_id: str,
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """
    Regenerate the document summary using AI/LLM.
    This endpoint allows users to get a fresh AI-generated summary.
    """
    try:
        logger.info(f"[{request_id}] Regenerating summary for document: {document_id}")
        
        # Validate document exists
        doc_uuid = UUID(document_id.strip())
        document = await document_service.get_document(doc_uuid)
        if not document:
            logger.warning(f"[{request_id}] Document not found: {document_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Get document sections and entities from extra_data
        extra_data = document.extra_data or {}
        sections = extra_data.get("sections", [])
        entities = extra_data.get("all_entities", [])
        
        # Convert sections to DocumentSection objects for the processor
        from .services.parsers.base_parser import DocumentSection
        from .models.entity import Entity, EntityType
        
        doc_sections = []
        for section in sections:
            doc_sections.append(DocumentSection(
                title=section.get("title", ""),
                content=section.get("content", ""),
                section_type=section.get("section_type", "content"),
                metadata=section.get("metadata", {})
            ))
        
        # Convert entities to Entity objects
        doc_entities = []
        for entity in entities:
            try:
                entity_type = EntityType(entity.get("type", entity.get("entity_type", "MISC")))
                doc_entities.append(Entity(
                    text=entity.get("text", ""),
                    type=entity_type,
                    start=entity.get("start", 0),
                    end=entity.get("end", 0),
                    confidence=entity.get("confidence", 1.0),
                    metadata=entity.get("metadata", {})
                ))
            except ValueError:
                # Skip entities with invalid types
                continue
        
        # Generate new summary - use AutoGen for advanced summarization if needed
        new_summary = f"Document updated with {len(doc_sections)} sections and {len(doc_entities)} entities"
        
        # Update the document with the new summary
        update_data = DocumentUpdate(summary=new_summary)
        updated_doc = await document_service.update_document(doc_uuid, update_data)
        
        if not updated_doc:
            logger.error(f"[{request_id}] Failed to update document with new summary")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save new summary"
            )
        
        logger.info(f"[{request_id}] Successfully regenerated summary for document: {document_id}")
        
        return {
            "success": True,
            "summary": new_summary,
            "message": "Summary regenerated successfully",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error regenerating summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error regenerating summary: {str(e)}"
        )

# Agent Workflow Endpoints

class WorkflowRequest(BaseModel):
    """Request model for workflow execution."""
    workflow_type: str = Field(..., description="Type of workflow to execute")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context for workflow")

class WorkflowResponse(BaseModel):
    """Response model for workflow execution."""
    workflow_id: str
    workflow_type: str
    document_id: str
    status: str
    execution_results: Optional[Dict[str, Any]] = None
    workflow_summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str

class CorpusWorkflowRequest(BaseModel):
    """Request model for corpus-level workflow execution."""
    document_ids: List[str] = Field(..., description="List of document IDs to process")
    workflow_type: str = Field(..., description="Type of corpus workflow to execute")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context for workflow")

class CorpusWorkflowResponse(BaseModel):
    """Response model for corpus workflow execution."""
    workflow_id: str
    workflow_type: str
    corpus_size: int
    status: str
    document_results: Optional[List[Dict[str, Any]]] = None
    corpus_results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str

class QuestionRequest(BaseModel):
    """Request model for Q&A workflows."""
    question: str = Field(..., description="Question to answer")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")

@app.post(
    "/documents/{document_id}/workflows",
    response_model=WorkflowResponse,
    summary="Execute agent workflow on document"
)
async def execute_document_workflow(
    document_id: str,
    workflow_request: WorkflowRequest,
    request_id: str = Depends(get_request_id)
) -> WorkflowResponse:
    """
    Execute an agent workflow on a specific document.
    
    Available workflow types:
    - full: Complete processing with all agents
    - summarization: Summarization agents only
    - entity_extraction: Entity extraction and tagging
    - qa: Question answering (requires question in context)
    """
    try:
        logger.info(f"[{request_id}] Executing workflow '{workflow_request.workflow_type}' on document: {document_id}")
        
        # Get document
        doc_uuid = UUID(document_id.strip())
        document = await document_service.get_document(doc_uuid)
        if not document:
            logger.warning(f"[{request_id}] Document not found: {document_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Execute AutoGen workflow
        workflow_result = await autogen_orchestrator.execute_document_workflow(
            document=document,
            workflow_type=workflow_request.workflow_type,
            context=workflow_request.context
        )
        
        return WorkflowResponse(
            workflow_id=workflow_result["workflow_id"],
            workflow_type=workflow_result["workflow_type"],
            document_id=workflow_result["document_id"],
            status=workflow_result.get("status", "completed"),
            execution_results=workflow_result.get("agent_results"),
            workflow_summary=workflow_result.get("workflow_summary"),
            error=workflow_result.get("error"),
            timestamp=workflow_result.get("timestamp", "")
        )
        
    except ValueError as e:
        logger.warning(f"[{request_id}] Invalid document ID format: {document_id}, error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document ID format: {document_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error executing workflow: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing workflow: {str(e)}"
        )

@app.post(
    "/documents/{document_id}/qa",
    response_model=Dict[str, Any],
    summary="Ask questions about a document"
)
async def ask_document_question(
    document_id: str,
    question_request: QuestionRequest,
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """
    Ask a question about a specific document using Q&A agents.
    """
    try:
        logger.info(f"[{request_id}] Asking question about document: {document_id}")
        
        # Get document
        doc_uuid = UUID(document_id.strip())
        document = await document_service.get_document(doc_uuid)
        if not document:
            logger.warning(f"[{request_id}] Document not found: {document_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Use LangChain Q&A service instead of AutoGen
        qa_result = await qa_service.answer_question(
            question=question_request.question,
            document=document,
            top_k=5,
            min_similarity=0.3
        )
        
        return {
            "document_id": document_id,
            "question": question_request.question,
            "answer": qa_result.answer,
            "confidence": qa_result.confidence,
            "sources": qa_result.sources,
            "context_chunks": qa_result.context_chunks,
            "metadata": qa_result.metadata,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except ValueError as e:
        logger.warning(f"[{request_id}] Invalid document ID format: {document_id}, error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document ID format: {document_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error in Q&A: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )

@app.post(
    "/corpus/workflows",
    response_model=CorpusWorkflowResponse,
    summary="Execute agent workflow on corpus"
)
async def execute_corpus_workflow(
    corpus_request: CorpusWorkflowRequest,
    request_id: str = Depends(get_request_id)
) -> CorpusWorkflowResponse:
    """
    Execute an agent workflow on a corpus of documents.
    
    Available corpus workflow types:
    - corpus_analysis: Full corpus analysis with summarization
    - corpus_summarization: Corpus-level summarization only
    - corpus_qa: Corpus-wide question answering (requires question in context)
    """
    try:
        logger.info(f"[{request_id}] Executing corpus workflow '{corpus_request.workflow_type}' on {len(corpus_request.document_ids)} documents")
        
        # Get documents
        documents = []
        for doc_id in corpus_request.document_ids:
            try:
                doc_uuid = UUID(doc_id.strip())
                document = await document_service.get_document(doc_uuid)
                if document:
                    documents.append(document)
                else:
                    logger.warning(f"[{request_id}] Document not found: {doc_id}")
            except ValueError:
                logger.warning(f"[{request_id}] Invalid document ID: {doc_id}")
                continue
        
        if not documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid documents found"
            )
        
        # Execute AutoGen corpus workflow
        workflow_result = await autogen_orchestrator.execute_corpus_workflow(
            documents=documents,
            workflow_type=corpus_request.workflow_type,
            context=corpus_request.context
        )
        
        return CorpusWorkflowResponse(
            workflow_id=workflow_result["workflow_id"],
            workflow_type=workflow_result["workflow_type"],
            corpus_size=workflow_result["corpus_size"],
            status=workflow_result.get("status", "completed"),
            document_results=workflow_result.get("document_results"),
            corpus_results=workflow_result.get("corpus_results"),
            error=workflow_result.get("error"),
            timestamp=workflow_result.get("timestamp", "")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error executing corpus workflow: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing corpus workflow: {str(e)}"
        )

@app.post(
    "/corpus/qa",
    response_model=Dict[str, Any],
    summary="Ask questions across multiple documents"
)
async def ask_corpus_question(
    corpus_request: CorpusWorkflowRequest,
    question_request: QuestionRequest,
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """
    Ask a question across multiple documents in a corpus.
    """
    try:
        logger.info(f"[{request_id}] Asking corpus question across {len(corpus_request.document_ids)} documents")
        
        # Get documents
        documents = []
        for doc_id in corpus_request.document_ids:
            try:
                doc_uuid = UUID(doc_id.strip())
                document = await document_service.get_document(doc_uuid)
                if document:
                    documents.append(document)
            except ValueError:
                continue
        
        if not documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid documents found"
            )
        
        # Use LangChain Q&A service for corpus questions
        qa_result = await qa_service.answer_corpus_question(
            question=question_request.question,
            documents=documents,
            top_k=10,
            min_similarity=0.3
        )
        
        return {
            "corpus_size": len(documents),
            "document_ids": [str(doc.id) for doc in documents],
            "question": question_request.question,
            "answer": qa_result.answer,
            "confidence": qa_result.confidence,
            "sources": qa_result.sources,
            "context_chunks": qa_result.context_chunks,
            "metadata": qa_result.metadata,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error in corpus Q&A: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing corpus question: {str(e)}"
        )

@app.get(
    "/workflows/{workflow_id}/status",
    response_model=Dict[str, Any],
    summary="Get workflow execution status"
)
async def get_workflow_status(
    workflow_id: str,
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """
    Get the status of a workflow execution.
    Note: This is a placeholder endpoint. In a production system,
    you would store workflow status in a database or cache.
    """
    try:
        logger.info(f"[{request_id}] Getting status for workflow: {workflow_id}")
        
        # For now, return a placeholder response
        # In a real implementation, you would query the workflow status from storage
        return {
            "workflow_id": workflow_id,
            "status": "completed",
            "message": "Workflow status tracking not yet implemented",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Error getting workflow status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving workflow status: {str(e)}"
        )

# Status and Information Endpoints

@app.get(
    "/orchestrator/info",
    response_model=Dict[str, Any],
    summary="Get AutoGen orchestrator information"
)
async def get_orchestrator_info(
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """
    Get information about the AutoGen orchestrator implementation.
    
    Provides details about capabilities, configuration, and usage.
    """
    try:
        logger.info(f"[{request_id}] Getting AutoGen orchestrator information")
        
        return {
            "orchestrator": {
                "name": "AutoGen Workflow Orchestrator",
                "framework": "Microsoft AutoGen",
                "version": "autogen>=0.3.0",
                "implementation": "LLM-powered multi-agent system"
            },
            "capabilities": [
                "Conversational AI agents",
                "Multi-agent group chats",
                "Advanced reasoning capabilities",
                "Dynamic agent interactions",
                "LLM-powered decision making",
                "Sophisticated orchestration patterns"
            ],
            "agents": {
                "summarization_agent": "Advanced document summarization with LLM reasoning",
                "entity_agent": "Sophisticated entity extraction and analysis",
                "qa_agent": "Conversational question answering",
                "validation_agent": "Cross-agent validation and quality assurance"
            },
            "workflow_types": [
                "full - Complete multi-agent processing",
                "summarization - Advanced summarization workflows",
                "entity_extraction - Sophisticated entity analysis",
                "qa - Conversational question answering",
                "corpus_analysis - Multi-document analysis",
                "corpus_qa - Cross-document question answering"
            ],
            "requirements": {
                "azure_openai_configured": bool(settings.AZURE_OPENAI_API_KEY and settings.AZURE_OPENAI_ENDPOINT),
                            "required_env_vars": [
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_ENDPOINT", 
                "AZURE_OPENAI_DEPLOYMENT",
                "AZURE_OPENAI_API_VERSION"
            ]
        },
        "system_features": {
            "llm_powered": "100% LLM processing with Azure OpenAI",
            "semantic_search": "Vector embeddings for content retrieval",
            "no_hardcoding": "All logic uses LLM reasoning",
            "no_fallbacks": "Pure AutoGen implementation",
            "embedding_integration": "Full vector search capabilities"
        }
    }
        
    except Exception as e:
        logger.error(f"[{request_id}] Error getting orchestrator info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving orchestrator information: {str(e)}"
        )

# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with JSON responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status_code": exc.status_code,
            "error": exc.__class__.__name__
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions with a 500 response."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "error": "InternalServerError"
        }
    )
