"""LangChain-based Q&A service using embeddings and retrieval."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain.schema import Document as LangChainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .embedding_service import get_embedding_service
from ..models.document import Document

logger = logging.getLogger(__name__)


@dataclass
class QAResult:
    """Result from Q&A query."""
    question: str
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    context_chunks: List[str]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Result from semantic search."""
    document_id: str
    document_filename: str
    section_title: str
    section_content: str
    chunk_text: str
    similarity_score: float
    chunk_id: int = 0
    metadata: Optional[Dict[str, Any]] = None


class LangChainQA:
    """LangChain-based Q&A service using embeddings and semantic search."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.embedding_service = get_embedding_service(config)
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size", 1000),
            chunk_overlap=self.config.get("chunk_overlap", 200),
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # In-memory document index for semantic search
        self.document_index: Dict[str, List[Dict[str, Any]]] = {}
    
    async def answer_question(
        self, 
        question: str, 
        document: Document,
        top_k: int = 5,
        min_similarity: float = 0.3
    ) -> QAResult:
        """
        Answer a question about a document using LangChain embeddings and retrieval.
        
        Args:
            question: The question to answer
            document: The document to search in
            top_k: Number of relevant chunks to retrieve
            min_similarity: Minimum similarity threshold for chunks
            
        Returns:
            QAResult with answer and sources
        """
        try:
            # Get relevant chunks using semantic search
            relevant_chunks = await self._find_relevant_chunks(
                question, document, top_k, min_similarity
            )
            
            if not relevant_chunks:
                return QAResult(
                    question=question,
                    answer="I couldn't find relevant information in the document to answer your question.",
                    confidence=0.0,
                    sources=[],
                    context_chunks=[]
                )
            
            # Generate answer using the most relevant chunks
            answer, confidence = await self._generate_answer(question, relevant_chunks)
            
            # Prepare sources
            sources = []
            context_chunks = []
            for chunk in relevant_chunks:
                sources.append({
                    "chunk_id": chunk.chunk_id,
                    "section_title": chunk.section_title,
                    "similarity_score": chunk.similarity_score,
                    "document_filename": chunk.document_filename
                })
                context_chunks.append(chunk.chunk_text)
            
            return QAResult(
                question=question,
                answer=answer,
                confidence=confidence,
                sources=sources,
                context_chunks=context_chunks,
                metadata={
                    "method": "langchain_embedding_search",
                    "chunks_retrieved": len(relevant_chunks),
                    "avg_similarity": sum(c.similarity_score for c in relevant_chunks) / len(relevant_chunks)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in LangChain Q&A: {str(e)}")
            return QAResult(
                question=question,
                answer=f"I encountered an error while processing your question: {str(e)}",
                confidence=0.0,
                sources=[],
                context_chunks=[]
            )
    
    async def _find_relevant_chunks(
        self, 
        question: str, 
        document: Document,
        top_k: int,
        min_similarity: float
    ) -> List[SearchResult]:
        """Find relevant chunks using semantic search."""
        try:
            # Ensure document is indexed
            await self._index_document(document)
            
            # Get document chunks
            doc_chunks = self.document_index.get(str(document.id), [])
            if not doc_chunks:
                logger.warning(f"No chunks found for document {document.id}")
                return []
            
            # Generate embedding for the question
            question_embedding = await self.embedding_service.embed_text(question)
            if not question_embedding:
                logger.error("Failed to generate question embedding")
                return []
            
            # Calculate similarities
            similarities = []
            for chunk in doc_chunks:
                chunk_embedding = chunk.get("embedding", [])
                if chunk_embedding:
                    similarity = self.embedding_service.cosine_similarity(question_embedding, chunk_embedding)
                    if similarity >= min_similarity:
                        similarities.append((chunk, similarity))
            
            # Sort by similarity and take top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_results = similarities[:top_k]
            
            # Convert to SearchResult objects
            search_results = []
            for chunk, similarity in top_results:
                result = SearchResult(
                    document_id=str(document.id),
                    document_filename=document.filename,
                    section_title=chunk.get("section_title", ""),
                    section_content=chunk.get("section_content", ""),
                    chunk_text=chunk.get("text", ""),
                    similarity_score=similarity,
                    chunk_id=chunk.get("chunk_id", 0),
                    metadata=chunk.get("metadata", {})
                )
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error finding relevant chunks: {str(e)}")
            return []
    
    async def _generate_answer(
        self, 
        question: str, 
        relevant_chunks: List[SearchResult]
    ) -> Tuple[str, float]:
        """
        Generate an answer based on relevant chunks.
        This is a simple implementation - in a full LangChain setup,
        you would use a proper LLM chain here.
        """
        try:
            # Combine relevant chunks
            context = "\n\n".join([
                f"Section: {chunk.section_title}\nContent: {chunk.chunk_text}"
                for chunk in relevant_chunks
            ])
            
            # Simple answer generation based on similarity scores and content
            # In a real implementation, you would use LangChain's RetrievalQA chain
            # with a proper LLM for more sophisticated answer generation
            
            # Calculate confidence based on similarity scores
            avg_similarity = sum(chunk.similarity_score for chunk in relevant_chunks) / len(relevant_chunks)
            confidence = min(avg_similarity * 1.2, 1.0)  # Boost confidence slightly
            
            # Generate a simple answer
            if len(relevant_chunks) == 1:
                answer = f"Based on the document content, {relevant_chunks[0].chunk_text[:200]}..."
            else:
                # Combine information from multiple chunks
                key_info = []
                for chunk in relevant_chunks[:3]:  # Use top 3 chunks
                    key_info.append(chunk.chunk_text[:100])
                answer = f"Based on the document content, the relevant information includes: {' '.join(key_info)}..."
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I couldn't generate a proper answer from the retrieved information.", 0.0
    
    async def answer_corpus_question(
        self, 
        question: str, 
        documents: List[Document],
        top_k: int = 10,
        min_similarity: float = 0.3
    ) -> QAResult:
        """
        Answer a question across multiple documents.
        
        Args:
            question: The question to answer
            documents: List of documents to search in
            top_k: Number of relevant chunks to retrieve
            min_similarity: Minimum similarity threshold for chunks
            
        Returns:
            QAResult with answer and sources
        """
        try:
            # Index all documents
            for doc in documents:
                await self._index_document(doc)
            
            # Search across all documents
            search_results = await self._search_across_documents(
                question, documents, top_k, min_similarity
            )
            
            if not search_results:
                return QAResult(
                    question=question,
                    answer="I couldn't find relevant information across the documents to answer your question.",
                    confidence=0.0,
                    sources=[],
                    context_chunks=[]
                )
            
            # Generate answer using the most relevant chunks
            answer, confidence = await self._generate_answer(question, search_results)
            
            # Prepare sources
            sources = []
            context_chunks = []
            for chunk in search_results:
                sources.append({
                    "chunk_id": chunk.chunk_id,
                    "section_title": chunk.section_title,
                    "similarity_score": chunk.similarity_score,
                    "document_filename": chunk.document_filename,
                    "document_id": chunk.document_id
                })
                context_chunks.append(chunk.chunk_text)
            
            return QAResult(
                question=question,
                answer=answer,
                confidence=confidence,
                sources=sources,
                context_chunks=context_chunks,
                metadata={
                    "method": "langchain_corpus_search",
                    "documents_searched": len(documents),
                    "chunks_retrieved": len(search_results),
                    "avg_similarity": sum(c.similarity_score for c in search_results) / len(search_results)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in corpus Q&A: {str(e)}")
            return QAResult(
                question=question,
                answer=f"I encountered an error while processing your question across documents: {str(e)}",
                confidence=0.0,
                sources=[],
                context_chunks=[]
            )
    
    async def _index_document(self, document: Document) -> bool:
        """Index a document for semantic search."""
        try:
            doc_id = str(document.id)
            
            # Check if already indexed
            if doc_id in self.document_index:
                return True
            
            # Get document sections
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
            doc_chunks = []
            for section in embedded_sections:
                section_title = section.get("title", "")
                section_content = section.get("content", "")
                
                for chunk in section.get("chunks", []):
                    chunk_data = {
                        "document_id": doc_id,
                        "document_filename": document.filename,
                        "section_title": section_title,
                        "section_content": section_content,
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "embedding": chunk["embedding"],
                        "word_count": chunk["word_count"],
                        "metadata": {
                            "document_id": doc_id,
                            "section_title": section_title
                        }
                    }
                    doc_chunks.append(chunk_data)
            
            # Store in document index
            self.document_index[doc_id] = doc_chunks
            
            logger.info(f"Indexed document {document.filename} with {len(doc_chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index document {document.id}: {str(e)}")
            return False
    
    async def _search_across_documents(
        self, 
        question: str, 
        documents: List[Document],
        top_k: int,
        min_similarity: float
    ) -> List[SearchResult]:
        """Search across multiple documents."""
        try:
            # Generate embedding for the question
            question_embedding = await self.embedding_service.embed_text(question)
            if not question_embedding:
                logger.error("Failed to generate question embedding")
                return []
            
            # Collect all chunks from all documents
            all_chunks = []
            for doc in documents:
                doc_chunks = self.document_index.get(str(doc.id), [])
                all_chunks.extend(doc_chunks)
            
            if not all_chunks:
                return []
            
            # Calculate similarities
            similarities = []
            for chunk in all_chunks:
                chunk_embedding = chunk.get("embedding", [])
                if chunk_embedding:
                    similarity = self.embedding_service.cosine_similarity(question_embedding, chunk_embedding)
                    if similarity >= min_similarity:
                        similarities.append((chunk, similarity))
            
            # Sort by similarity and take top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_results = similarities[:top_k]
            
            # Convert to SearchResult objects
            search_results = []
            for chunk, similarity in top_results:
                result = SearchResult(
                    document_id=chunk["document_id"],
                    document_filename=chunk["document_filename"],
                    section_title=chunk.get("section_title", ""),
                    section_content=chunk.get("section_content", ""),
                    chunk_text=chunk.get("text", ""),
                    similarity_score=similarity,
                    chunk_id=chunk.get("chunk_id", 0),
                    metadata=chunk.get("metadata", {})
                )
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching across documents: {str(e)}")
            return []


# Global Q&A service instance
_qa_service = None


def get_qa_service(config: Optional[Dict[str, Any]] = None) -> LangChainQA:
    """Get the global Q&A service instance."""
    global _qa_service
    if _qa_service is None:
        _qa_service = LangChainQA(config)
    return _qa_service
