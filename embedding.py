# pip install sentence-transformers

from sentence_transformers import SentenceTransformer
from typing import List, Optional
import threading


class EmbeddingService:
    """Singleton service for handling text embeddings"""
    
    _instance: Optional['EmbeddingService'] = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    print("Initializing SentenceTransformer model...")
                    # Force CPU usage to avoid CUDA compatibility issues
                    self.model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
                    print("SentenceTransformer model initialized successfully!")
                    self._initialized = True
    
    def get_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for the given text"""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        embeddings = self.model.encode(text).tolist()
        return embeddings
    
    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise ValueError("Input must be a list of strings")
        
        embeddings = self.model.encode(texts).tolist()
        return embeddings


# Global instance - will be initialized once
embedding_service = EmbeddingService()