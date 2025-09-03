"""Service for document storage, retrieval, and processing."""
import asyncio
import json
import shutil
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, BinaryIO
from uuid import UUID, uuid4

from ..core.config import settings
from ..models.document import Document, DocumentStatus, DocumentUpdate, DocumentCreate
from ..services.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for managing documents, including storage, retrieval, and processing."""
    
    def __init__(self, data_dir: str = "./data"):
        """Initialize the document service."""
        self.data_dir = Path(data_dir)
        self.upload_dir = self.data_dir / "uploads"
        self.documents_file = self.data_dir / "documents.json"
        self.documents: Dict[UUID, Document] = {}
        self.processor = DocumentProcessor()
        
        # Ensure directories exist
        self._ensure_dirs()
        self._load_documents()
    
    def _ensure_dirs(self) -> None:
        """Ensure required directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(exist_ok=True)
        
        if not self.documents_file.exists():
            self.documents_file.write_text("{}")
    
    def _load_documents(self) -> None:
        """Load documents from the JSON file."""
        try:
            data = json.loads(self.documents_file.read_text())
            self.documents = {}
            logger.info(f"Loading {len(data)} documents from {self.documents_file}")
            for k, v in data.items():
                # Use the key as the document ID
                doc_data = v.copy()
                doc_data['id'] = k  # Set the ID to the key
                # Create the document with the UUID as the key
                self.documents[UUID(k)] = Document(**doc_data)
            logger.info(f"Successfully loaded {len(self.documents)} documents")
        except (json.JSONDecodeError, FileNotFoundError, ValueError) as e:
            logger.warning(f"Error loading documents: {e}")
            self.documents = {}
    
    def _save_documents(self) -> None:
        """Save documents to the JSON file."""
        # Convert UUIDs to strings for JSON serialization
        data = {str(k): v.model_dump() for k, v in self.documents.items()}
        self.documents_file.write_text(json.dumps(data, indent=2, default=str))
    
    async def save_uploaded_file(
        self, 
        file: Union[BinaryIO, bytes], 
        filename: str,
        content_type: Optional[str] = None
    ) -> Tuple[Path, str]:
        """
        Save an uploaded file to the uploads directory.
        
        Args:
            file: The file content as bytes or a file-like object
            filename: The original filename
            content_type: The MIME type of the file
            
        Returns:
            Tuple of (saved file path, content type)
        """
        # Ensure filename is safe
        safe_filename = "".join(c for c in filename if c.isalnum() or c in ' .-_,')
        if not safe_filename:
            safe_filename = str(uuid4())
            
        # Ensure unique filename
        file_path = self.upload_dir / safe_filename
        counter = 1
        while file_path.exists():
            name_parts = file_path.stem.split('_')
            if name_parts[-1].isdigit() and len(name_parts) > 1:
                base = '_'.join(name_parts[:-1])
                counter = int(name_parts[-1]) + 1
            else:
                base = file_path.stem
                counter += 1
            file_path = file_path.with_name(f"{base}_{counter}{file_path.suffix}")
        
        # Save the file
        if isinstance(file, bytes):
            file_path.write_bytes(file)
        else:
            file_path.write_bytes(file.read())
        
        # Try to determine content type if not provided
        if not content_type:
            import mimetypes
            content_type, _ = mimetypes.guess_type(file_path)
            if not content_type:
                content_type = 'application/octet-stream'
        
        return file_path, content_type
    
    async def create_document(self, document_data: DocumentCreate, file_content: bytes) -> Document:
        """
        Create a new document and start processing it.
        
        Args:
            document_data: Document creation data
            file_content: The file content as bytes
            
        Returns:
            The created Document instance
        """
        logger.debug(f"Creating document with data: {document_data.model_dump()}")
        logger.debug(f"File content length: {len(file_content)}")
        
        # Save the uploaded file
        file_path, content_type = await self.save_uploaded_file(
            file_content,
            filename=document_data.filename,
            content_type=document_data.content_type
        )
        logger.debug(f"File saved to: {file_path}, content_type: {content_type}")
        
        # Create the document with the correct fields
        document = Document(
            filename=document_data.filename,
            content_type=content_type,
            size=len(file_content),
            status=DocumentStatus.UPLOADED,
            metadata={
                "original_filename": document_data.filename,
                "stored_path": str(file_path),
                **(document_data.metadata or {})
            }
        )
        logger.debug(f"Document object created: {document.model_dump()}")
        
        # Save the document
        self.documents[document.id] = document
        self._save_documents()
        logger.debug(f"Document saved to memory and disk with ID: {document.id}")
        logger.debug(f"Total documents in memory: {len(self.documents)}")
        
        # Start processing in background
        try:
            # Process the document immediately
            await self._process_document(document.id, file_path)
            logger.debug(f"Document processing completed for {document.id}")
        except Exception as e:
            logger.error(f"Error processing document {document.id}: {str(e)}", exc_info=True)
            # Update document status to failed
            document.status = DocumentStatus.FAILED
            document.error = str(e)
            document.updated_at = datetime.now(timezone.utc)
            self._save_documents()
        
        return document
    
    async def _process_document(self, doc_id: UUID, file_path: Path) -> None:
        """
        Process a document through the pipeline.
        
        Args:
            doc_id: The document ID
            file_path: Path to the document file
        """
        if doc_id not in self.documents:
            logger.error(f"Document {doc_id} not found for processing")
            return
            
        document = self.documents[doc_id]
        
        try:
            # Update status to processing
            document.status = DocumentStatus.PROCESSING
            document.updated_at = datetime.now(timezone.utc)
            self._save_documents()
            
            # Process the document
            processed_doc = await self.processor.process_document(
                file_path=file_path,
                filename=document.filename,
                content_type=document.content_type
            )
            
            # Update the document with processed data
            document.status = processed_doc.status
            document.summary = processed_doc.summary
            document.metadata.update(processed_doc.metadata or {})
            document.extra_data = processed_doc.extra_data
            document.updated_at = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {str(e)}", exc_info=True)
            document.status = DocumentStatus.FAILED
            document.error = str(e)
            document.updated_at = datetime.now(timezone.utc)
            
        finally:
            self._save_documents()
        
        # Store the document
        self.documents[doc_id] = document
        self._save_documents()
        
        return document
    
    async def get_document(self, doc_id: UUID) -> Optional[Document]:
        """
        Get a document by ID.
        
        Args:
            doc_id: The document ID
            
        Returns:
            The Document instance, or None if not found
        """
        logger.info(f"Looking for document with ID: {doc_id}")
        logger.info(f"Available document IDs: {list(self.documents.keys())}")
        logger.info(f"Total documents loaded: {len(self.documents)}")
        document = self.documents.get(doc_id)
        if document:
            logger.info(f"Found document: {document.filename}")
        else:
            logger.warning(f"Document not found: {doc_id}")
            # Check if the document exists as a string key
            str_doc_id = str(doc_id)
            logger.info(f"Checking if document exists as string key: {str_doc_id}")
            if str_doc_id in {str(k) for k in self.documents.keys()}:
                logger.warning(f"Document exists as string but not as UUID object")
        return document
    
    async def update_document(
        self, 
        doc_id: UUID, 
        update: DocumentUpdate
    ) -> Optional[Document]:
        """
        Update a document.
        
        Args:
            doc_id: The document ID
            update: The update data
            
        Returns:
            The updated Document, or None if not found
        """
        if doc_id not in self.documents:
            return None
            
        document = self.documents[doc_id]
        
        # Update fields
        if update.status:
            document.status = update.status
        if update.summary is not None:
            document.summary = update.summary
        if update.metadata:
            document.metadata.update(update.metadata)
        if update.extra_data:
            if document.extra_data is None:
                document.extra_data = {}
            document.extra_data.update(update.extra_data)
            
        document.updated_at = datetime.now(timezone.utc)
        self._save_documents()
        
        return document
    
    async def list_documents(
        self, 
        status: Optional[DocumentStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Document]:
        """
        List documents, optionally filtered by status.
        
        Args:
            status: Optional status filter
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            List of matching documents
        """
        docs = list(self.documents.values())
        
        if status is not None:
            docs = [d for d in docs if d.status == status]
            
        # Sort by creation date, newest first
        docs.sort(key=lambda d: d.created_at, reverse=True)
        
        return docs[offset:offset+limit]


# Create a singleton instance
document_service = DocumentService()
