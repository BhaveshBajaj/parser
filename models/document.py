"""Document models for the document processing system."""
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict


class DocumentStatus(str, Enum):
    """Status of a document in the processing pipeline."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    SUMMARIZED = "summarized"
    PROCESSED = "processed"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentBase(BaseModel):
    """Base document model."""
    filename: str = Field(..., description="Original filename")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the document"
    )


class DocumentCreate(BaseModel):
    """Model for creating a new document."""
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type of the file")
    size: int = Field(..., description="File size in bytes")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the document"
    )


class Document(DocumentBase):
    """Document model with system fields and processing results."""
    content_type: str = Field(..., description="MIME type of the file")
    size: int = Field(..., description="File size in bytes")
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique document identifier"
    )
    status: DocumentStatus = Field(
        default=DocumentStatus.UPLOADED,
        description="Current processing status of the document"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the document was created"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the document was last updated"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Generated summary of the document"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if processing failed"
    )
    extra_data: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional data from processing (sections, entities, etc.)"
    )

    # Pydantic v2 config
    model_config = ConfigDict(
        json_encoders={
            UUID: str,
            datetime: lambda dt: dt.isoformat(),
        }
    )


class DocumentUpdate(BaseModel):
    """Model for updating a document."""
    status: Optional[DocumentStatus] = Field(
        default=None,
        description="New status of the document"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Updated summary of the document"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if processing failed"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata to merge with existing metadata"
    )
    extra_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional data to merge with existing extra_data"
    )
