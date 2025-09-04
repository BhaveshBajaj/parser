"""Embedding service for document vector representations using LangChain and HuggingFace."""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import hashlib
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing document embeddings using LangChain and HuggingFace."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model_name = self.config.get("embedding_model", "sentence-transformers/all-mpnet-base-v2")
        self.chunk_size = self.config.get("chunk_size", 512)
        self.chunk_overlap = self.config.get("chunk_overlap", 50)
        self.cache_embeddings = self.config.get("cache_embeddings", True)
        self.cache_dir = Path(self.config.get("cache_dir", "data/embeddings"))
        
        # Create cache directory if it doesn't exist
        if self.cache_embeddings:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LangChain components
        self._embeddings = None
        self._text_splitter = None
        self._initialize_langchain()
    
    def _initialize_langchain(self):
        """Initialize LangChain components for embeddings and text splitting."""
        try:
            # Initialize HuggingFace embeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize text splitter
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            logger.info(f"LangChain components initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LangChain components: {str(e)}")
            self._embeddings = None
            self._text_splitter = None
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text using LangChain HuggingFace embeddings."""
        if not text or not text.strip():
            return []
        
        # Check cache first
        if self.cache_embeddings:
            cached_embedding = self._get_cached_embedding(text)
            if cached_embedding is not None:
                return cached_embedding
        
        try:
            if not self._embeddings:
                raise Exception("LangChain embeddings not initialized")
            
            # Use LangChain HuggingFace embeddings
            embedding = await self._get_langchain_embedding(text)
            
            # Cache the embedding
            if self.cache_embeddings and embedding:
                self._cache_embedding(text, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise Exception(f"Embedding generation failed: {str(e)}")
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using LangChain batch processing."""
        if not texts:
            return []
        
        try:
            if not self._embeddings:
                raise Exception("LangChain embeddings not initialized")
            
            # Use LangChain batch embedding for efficiency
            embeddings = await self._get_langchain_embeddings_batch(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {str(e)}")
            raise Exception(f"Batch embedding generation failed: {str(e)}")
    
    async def _get_langchain_embedding(self, text: str) -> List[float]:
        """Get embedding using LangChain HuggingFace embeddings."""
        try:
            # Use asyncio to run the synchronous embedding in a thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, self._embeddings.embed_query, text)
            return embedding
        except Exception as e:
            logger.error(f"LangChain HuggingFace embedding failed: {str(e)}")
            raise
    
    async def _get_langchain_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts using LangChain batch processing."""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(None, self._embeddings.embed_documents, texts)
            return embeddings
        except Exception as e:
            logger.error(f"LangChain batch embedding failed: {str(e)}")
            raise
    
    
    def _get_text_hash(self, text: str) -> str:
        """Generate a hash for text caching."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding if available."""
        try:
            text_hash = self._get_text_hash(text)
            cache_file = self.cache_dir / f"{text_hash}.json"
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return data.get('embedding')
        except Exception as e:
            logger.warning(f"Failed to load cached embedding: {str(e)}")
        
        return None
    
    def _cache_embedding(self, text: str, embedding: List[float]):
        """Cache an embedding."""
        try:
            text_hash = self._get_text_hash(text)
            cache_file = self.cache_dir / f"{text_hash}.json"
            
            cache_data = {
                'text_preview': text[:100] + "..." if len(text) > 100 else text,
                'embedding': embedding,
                'model': self.model_name,
                'timestamp': str(np.datetime64('now'))
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {str(e)}")
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into chunks using LangChain's RecursiveCharacterTextSplitter."""
        if not text or not text.strip():
            return []
        
        try:
            if self._text_splitter:
                # Use LangChain text splitter
                chunks = self._text_splitter.split_text(text)
                
                # Convert to our format
                chunk_objects = []
                current_pos = 0
                
                for i, chunk_text in enumerate(chunks):
                    chunk_objects.append({
                        'id': i,
                        'text': chunk_text.strip(),
                        'start_pos': current_pos,
                        'word_count': len(chunk_text.split())
                    })
                    current_pos += len(chunk_text)
                
                return chunk_objects
            else:
                raise Exception("LangChain text splitter not initialized")
        except Exception as e:
            logger.error(f"LangChain text splitting failed: {str(e)}")
            raise Exception(f"Text splitting failed: {str(e)}")
    
    
    async def embed_document_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for document sections."""
        embedded_sections = []
        
        for section in sections:
            title = section.get('title', '')
            content = section.get('content', '')
            
            # Combine title and content for embedding
            full_text = f"{title}\n{content}" if title else content
            
            if not full_text.strip():
                continue
            
            # Chunk the section if it's too long
            chunks = self.chunk_text(full_text)
            
            section_embeddings = []
            for chunk in chunks:
                embedding = await self.embed_text(chunk['text'])
                if embedding:
                    section_embeddings.append({
                        'chunk_id': chunk['id'],
                        'text': chunk['text'],
                        'embedding': embedding,
                        'word_count': chunk['word_count']
                    })
            
            if section_embeddings:
                embedded_sections.append({
                    'title': title,
                    'content': content,
                    'chunks': section_embeddings,
                    'section_embedding': await self._get_section_embedding(section_embeddings)
                })
        
        return embedded_sections
    
    async def _get_section_embedding(self, chunk_embeddings: List[Dict[str, Any]]) -> List[float]:
        """Generate a section-level embedding by averaging chunk embeddings."""
        if not chunk_embeddings:
            return []
        
        # Weight by word count and average
        total_words = sum(chunk['word_count'] for chunk in chunk_embeddings)
        if total_words == 0:
            return []
        
        embedding_dim = len(chunk_embeddings[0]['embedding'])
        weighted_sum = [0.0] * embedding_dim
        
        for chunk in chunk_embeddings:
            weight = chunk['word_count'] / total_words
            embedding = chunk['embedding']
            for i in range(embedding_dim):
                weighted_sum[i] += embedding[i] * weight
        
        return weighted_sum
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if not embedding1 or not embedding2 or len(embedding1) != len(embedding2):
            return 0.0
        
        # Convert to numpy arrays for easier computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def find_similar_sections(
        self, 
        query: str, 
        embedded_sections: List[Dict[str, Any]], 
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Find sections most similar to a query."""
        query_embedding = await self.embed_text(query)
        if not query_embedding:
            return []
        
        similarities = []
        
        for section in embedded_sections:
            section_embedding = section.get('section_embedding', [])
            if section_embedding:
                similarity = self.cosine_similarity(query_embedding, section_embedding)
                similarities.append((section, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    async def find_similar_chunks(
        self, 
        query: str, 
        embedded_sections: List[Dict[str, Any]], 
        top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Find individual chunks most similar to a query."""
        query_embedding = await self.embed_text(query)
        if not query_embedding:
            return []
        
        similarities = []
        
        for section in embedded_sections:
            for chunk in section.get('chunks', []):
                chunk_embedding = chunk.get('embedding', [])
                if chunk_embedding:
                    similarity = self.cosine_similarity(query_embedding, chunk_embedding)
                    chunk_with_section = {
                        **chunk,
                        'section_title': section.get('title', ''),
                        'section_content': section.get('content', '')
                    }
                    similarities.append((chunk_with_section, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# Global embedding service instance
_embedding_service = None


def get_embedding_service(config: Optional[Dict[str, Any]] = None) -> EmbeddingService:
    """Get the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(config)
    return _embedding_service


