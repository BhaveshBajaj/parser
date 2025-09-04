"""API routes for the document processing service."""
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

from config.settings import settings
from models.document import Document, DocumentCreate, DocumentStatus, DocumentUpdate
from services.document_service import document_service
from services.parsers import DocumentSection
from models.entity import Entity, EntityType
from services.agents.autogen_agents import AutoGenWorkflowOrchestrator
from services.langchain_qa import get_qa_service

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

# Entity extraction models
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

# Workflow models
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

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
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

    # Document endpoints
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

    # Add more endpoints here...
    # (The rest of the endpoints would be added in the same pattern)

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

    return app
