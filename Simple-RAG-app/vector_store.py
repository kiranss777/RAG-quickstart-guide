from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec
import time

class VectorStore:
    def __init__(self, api_key: str, index_name: str = "ads7390ragapp"):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = 384
        
        # Check if index exists, if not create it
        existing_indexes = self.pc.list_indexes().names()
        
        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=self.dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            # Wait for index to be ready
            while not self.pc.describe_index(index_name).status['ready']:
                time.sleep(1)
        
        # Connect to the index
        self.index = self.pc.Index(index_name)
    
    def upsert_chunks(self, chunks: List[Dict], embeddings: List[List[float]]):
        """Upsert chunks with embeddings to Pinecone"""
        vectors = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                'id': chunk['id'],
                'values': embedding,
                'metadata': {
                    'text': chunk['text'],
                    'method': chunk['method'],
                    'chunk_index': i
                }
            })
        
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch)
        
        print(f"✅ Upserted {len(vectors)} vectors to Pinecone index '{self.index_name}'")
    
    def query(self, query_embedding: List[float], top_k: int = 3) -> List[Dict]:
        """Query the index for similar chunks"""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        chunks = []
        for match in results['matches']:
            chunks.append({
                'id': match['id'],
                'text': match['metadata']['text'],
                'score': match['score'],
                'method': match['metadata']['method']
            })
        
        return chunks
    
    def delete_all(self):
        """Delete all vectors from the index"""
        self.index.delete(delete_all=True)
        print(f"🗑️ Cleared all vectors from index '{self.index_name}'")
    
    def get_stats(self):
        """Get index statistics"""
        stats = self.index.describe_index_stats()
        return stats