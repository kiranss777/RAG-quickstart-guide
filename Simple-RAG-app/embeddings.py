from typing import List
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self, pinecone_api_key: str = None):
        """Initialize with a 384-dimensional embedding model"""
        # Using all-MiniLM-L12-v2 which produces exactly 384 dimensions
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        print(f"✅ Loaded embedding model with {self.model.get_sentence_embedding_dimension()} dimensions")
        
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate 384-dimensional embeddings"""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def generate_embedding_single(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embedding = self.model.encode([text])[0]
        return embedding.tolist()