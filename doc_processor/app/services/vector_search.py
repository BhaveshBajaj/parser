"""Vector search utilities for semantic document search."""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import json
from pathlib import Path

from .embedding_service import get_embedding_service
from ..models.document import Document

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from vector search."""
    document_id: str
    document_filename: str
    section_title: str
    section_content: str
    chunk_text: str
    similarity_score: float
    chunk_id: int = 0
    metadata: Optional[Dict[str, Any]] = None


class VectorSearchIndex:
    """In-memory vector search index for documents."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.embedding_service = get_embedding_service(config)
        self.index_data = []  # List of indexed chunks with embeddings
        self.document_metadata = {}  # Document metadata cache
        
    async def add_document(self, document: Document) -> bool:
        """Add a document to the search index."""
        try:
            if not document.extra_data or "sections" not in document.extra_data:
                logger.warning(f"Document {document.id} has no sections to index")
                return False
            
            sections = document.extra_data["sections"]
            if not sections:
                return False
            
            # Generate embeddings for document sections
            embedded_sections = await self.embedding_service.embed_document_sections(sections)
            
            if not embedded_sections:
                logger.warning(f"Failed to generate embeddings for document {document.id}")
                return False
            
            # Add each chunk to the index
            for section in embedded_sections:
                section_title = section.get("title", "")
                section_content = section.get("content", "")
                
                for chunk in section.get("chunks", []):
                    chunk_data = {
                        "document_id": str(document.id),
                        "document_filename": document.filename,
                        "section_title": section_title,
                        "section_content": section_content,
                        "chunk_id": chunk["chunk_id"],
                        "chunk_text": chunk["text"],
                        "embedding": chunk["embedding"],
                        "word_count": chunk["word_count"]
                    }
                    self.index_data.append(chunk_data)
            
            # Store document metadata
            self.document_metadata[str(document.id)] = {
                "filename": document.filename,
                "file_type": document.file_type,
                "total_sections": len(sections),
                "total_chunks": sum(len(s.get("chunks", [])) for s in embedded_sections),
                "created_at": document.created_at.isoformat() if document.created_at else None
            }
            
            logger.info(f"Added document {document.filename} to search index with {len(embedded_sections)} sections")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document {document.id} to search index: {str(e)}")
            return False
    
    async def add_documents(self, documents: List[Document]) -> Dict[str, bool]:
        """Add multiple documents to the search index."""
        results = {}
        for document in documents:
            success = await self.add_document(document)
            results[str(document.id)] = success
        
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Added {successful}/{len(documents)} documents to search index")
        return results
    
    async def search(
        self, 
        query: str, 
        top_k: int = 10, 
        min_similarity: float = 0.0,
        document_filter: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Search for relevant chunks using vector similarity."""
        if not self.index_data:
            logger.warning("Search index is empty")
            return []
        
        try:
            # Generate embedding for the query
            query_embedding = await self.embedding_service.embed_text(query)
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            # Calculate similarities
            similarities = []
            for chunk_data in self.index_data:
                # Apply document filter if specified
                if document_filter and chunk_data["document_id"] not in document_filter:
                    continue
                
                chunk_embedding = chunk_data["embedding"]
                similarity = self.embedding_service.cosine_similarity(query_embedding, chunk_embedding)
                
                if similarity >= min_similarity:
                    similarities.append((chunk_data, similarity))
            
            # Sort by similarity and take top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_results = similarities[:top_k]
            
            # Convert to SearchResult objects
            search_results = []
            for chunk_data, similarity in top_results:
                result = SearchResult(
                    document_id=chunk_data["document_id"],
                    document_filename=chunk_data["document_filename"],
                    section_title=chunk_data["section_title"],
                    section_content=chunk_data["section_content"],
                    chunk_text=chunk_data["chunk_text"],
                    similarity_score=similarity,
                    chunk_id=chunk_data["chunk_id"],
                    metadata={
                        "word_count": chunk_data["word_count"],
                        "document_metadata": self.document_metadata.get(chunk_data["document_id"], {})
                    }
                )
                search_results.append(result)
            
            logger.info(f"Found {len(search_results)} results for query: {query[:50]}...")
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    async def search_by_document(
        self, 
        query: str, 
        document_ids: List[str], 
        top_k: int = 5
    ) -> Dict[str, List[SearchResult]]:
        """Search within specific documents."""
        results = {}
        
        for doc_id in document_ids:
            doc_results = await self.search(
                query=query,
                top_k=top_k,
                document_filter=[doc_id]
            )
            results[doc_id] = doc_results
        
        return results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the search index."""
        if not self.index_data:
            return {"total_chunks": 0, "total_documents": 0}
        
        document_ids = set(chunk["document_id"] for chunk in self.index_data)
        
        # Calculate statistics
        word_counts = [chunk["word_count"] for chunk in self.index_data]
        
        stats = {
            "total_chunks": len(self.index_data),
            "total_documents": len(document_ids),
            "avg_chunk_words": np.mean(word_counts) if word_counts else 0,
            "total_words": sum(word_counts),
            "documents": list(self.document_metadata.keys())
        }
        
        return stats
    
    def clear_index(self):
        """Clear the search index."""
        self.index_data.clear()
        self.document_metadata.clear()
        logger.info("Search index cleared")
    
    def remove_document(self, document_id: str) -> bool:
        """Remove a document from the search index."""
        try:
            # Remove chunks for this document
            original_count = len(self.index_data)
            self.index_data = [chunk for chunk in self.index_data if chunk["document_id"] != document_id]
            
            # Remove metadata
            if document_id in self.document_metadata:
                del self.document_metadata[document_id]
            
            removed_count = original_count - len(self.index_data)
            logger.info(f"Removed {removed_count} chunks for document {document_id}")
            return removed_count > 0
            
        except Exception as e:
            logger.error(f"Failed to remove document {document_id} from index: {str(e)}")
            return False
    
    async def save_index(self, filepath: Union[str, Path]):
        """Save the search index to disk."""
        try:
            index_data = {
                "index_data": self.index_data,
                "document_metadata": self.document_metadata,
                "config": self.config
            }
            
            with open(filepath, 'w') as f:
                json.dump(index_data, f, indent=2, default=str)
            
            logger.info(f"Search index saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save search index: {str(e)}")
            raise
    
    async def load_index(self, filepath: Union[str, Path]):
        """Load the search index from disk."""
        try:
            with open(filepath, 'r') as f:
                index_data = json.load(f)
            
            self.index_data = index_data.get("index_data", [])
            self.document_metadata = index_data.get("document_metadata", {})
            
            logger.info(f"Search index loaded from {filepath} with {len(self.index_data)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to load search index: {str(e)}")
            raise


class HybridSearch:
    """Hybrid search combining vector similarity with keyword matching."""
    
    def __init__(self, vector_index: VectorSearchIndex, config: Optional[Dict[str, Any]] = None):
        self.vector_index = vector_index
        self.config = config or {}
        self.vector_weight = self.config.get("vector_weight", 0.7)
        self.keyword_weight = self.config.get("keyword_weight", 0.3)
    
    def _keyword_score(self, query: str, text: str) -> float:
        """Calculate keyword matching score."""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words or not text_words:
            return 0.0
        
        intersection = query_words.intersection(text_words)
        return len(intersection) / len(query_words.union(text_words))
    
    async def search(
        self, 
        query: str, 
        top_k: int = 10, 
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
        """Perform hybrid search combining vector and keyword matching."""
        # Get vector search results
        vector_results = await self.vector_index.search(
            query=query,
            top_k=top_k * 2,  # Get more results for reranking
            min_similarity=min_similarity
        )
        
        if not vector_results:
            return []
        
        # Calculate hybrid scores
        hybrid_results = []
        for result in vector_results:
            # Calculate keyword score
            keyword_score = self._keyword_score(query, result.chunk_text)
            
            # Combine scores
            hybrid_score = (
                self.vector_weight * result.similarity_score + 
                self.keyword_weight * keyword_score
            )
            
            # Create new result with hybrid score
            hybrid_result = SearchResult(
                document_id=result.document_id,
                document_filename=result.document_filename,
                section_title=result.section_title,
                section_content=result.section_content,
                chunk_text=result.chunk_text,
                similarity_score=hybrid_score,
                chunk_id=result.chunk_id,
                metadata={
                    **(result.metadata or {}),
                    "vector_score": result.similarity_score,
                    "keyword_score": keyword_score,
                    "search_method": "hybrid"
                }
            )
            hybrid_results.append(hybrid_result)
        
        # Sort by hybrid score and return top k
        hybrid_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return hybrid_results[:top_k]


# Global search index instance
_search_index = None


def get_search_index(config: Optional[Dict[str, Any]] = None) -> VectorSearchIndex:
    """Get the global search index instance."""
    global _search_index
    if _search_index is None:
        _search_index = VectorSearchIndex(config)
    return _search_index


