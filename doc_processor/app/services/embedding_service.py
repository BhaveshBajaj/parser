"""Embedding service for document vector representations."""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing document embeddings."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model_name = self.config.get("embedding_model", "text-embedding-ada-002")
        self.chunk_size = self.config.get("chunk_size", 512)
        self.chunk_overlap = self.config.get("chunk_overlap", 50)
        self.cache_embeddings = self.config.get("cache_embeddings", True)
        self.cache_dir = Path(self.config.get("cache_dir", "data/embeddings"))
        
        # Create cache directory if it doesn't exist
        if self.cache_embeddings:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Azure OpenAI client
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client for embeddings."""
        try:
            from ..core.config import settings
            from openai import AzureOpenAI
            
            if settings.AZURE_OPENAI_API_KEY and settings.AZURE_OPENAI_ENDPOINT:
                self._client = AzureOpenAI(
                    api_key=settings.AZURE_OPENAI_API_KEY,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
                )
                logger.info("Azure OpenAI client initialized for embeddings")
            else:
                logger.warning("Azure OpenAI not configured - embeddings will use fallback method")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            self._client = None
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            return []
        
        # Check cache first
        if self.cache_embeddings:
            cached_embedding = self._get_cached_embedding(text)
            if cached_embedding is not None:
                return cached_embedding
        
        try:
            if self._client:
                # Use Azure OpenAI embeddings
                embedding = await self._get_azure_embedding(text)
            else:
                # Fallback to simple text-based embedding
                embedding = self._get_fallback_embedding(text)
            
            # Cache the embedding
            if self.cache_embeddings and embedding:
                self._cache_embedding(text, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            return self._get_fallback_embedding(text)
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = await self.embed_text(text)
            embeddings.append(embedding)
        return embeddings
    
    async def _get_azure_embedding(self, text: str) -> List[float]:
        """Get embedding using Azure OpenAI."""
        try:
            response = self._client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Azure OpenAI embedding failed: {str(e)}")
            raise
    
    def _get_fallback_embedding(self, text: str) -> List[float]:
        """Generate a simple fallback embedding based on text features."""
        # This is a very basic fallback - in production you'd want something better
        words = text.lower().split()
        
        # Create a simple feature vector
        features = []
        
        # Text length features
        features.extend([
            len(text) / 1000.0,  # Normalized text length
            len(words) / 100.0,  # Normalized word count
            len(set(words)) / len(words) if words else 0,  # Unique word ratio
        ])
        
        # Simple word frequency features (top 100 common words)
        common_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'what', 'when', 'where', 'why', 'how', 'who', 'which', 'all', 'any', 'some', 'many',
            'much', 'few', 'little', 'more', 'most', 'less', 'least', 'first', 'last', 'next',
            'previous', 'new', 'old', 'good', 'bad', 'big', 'small', 'long', 'short', 'high',
            'low', 'right', 'wrong', 'true', 'false', 'yes', 'no', 'not', 'only', 'also', 'even'
        ]
        
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Add frequency features for common words
        for word in common_words:
            features.append(word_counts.get(word, 0) / len(words) if words else 0)
        
        # Pad or truncate to fixed size (384 dimensions to match common embedding sizes)
        target_size = 384
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        # Normalize the vector
        norm = np.linalg.norm(features)
        if norm > 0:
            features = [f / norm for f in features]
        
        return features
    
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
        """Split text into chunks for embedding."""
        if not text or not text.strip():
            return []
        
        # Simple sentence-based chunking
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        current_length = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Add period back if it was removed by split
            if not sentence.endswith('.') and not sentence.endswith('!') and not sentence.endswith('?'):
                sentence += '.'
            
            sentence_length = len(sentence.split())
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append({
                    'id': chunk_id,
                    'text': current_chunk.strip(),
                    'start_pos': len(text) - len(' '.join(sentences[len(chunks):])),
                    'word_count': current_length
                })
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    overlap_words = current_chunk.split()[-self.chunk_overlap:]
                    current_chunk = ' '.join(overlap_words) + ' ' + sentence
                    current_length = len(overlap_words) + sentence_length
                else:
                    current_chunk = sentence
                    current_length = sentence_length
                
                chunk_id += 1
            else:
                if current_chunk:
                    current_chunk += ' ' + sentence
                else:
                    current_chunk = sentence
                current_length += sentence_length
        
        # Add final chunk if it has content
        if current_chunk.strip():
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'start_pos': 0,  # Approximate
                'word_count': current_length
            })
        
        return chunks
    
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
