from typing import List, Dict
import re
from sentence_transformers import SentenceTransformer
import numpy as np

class DocumentChunker:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def fixed_size_chunking(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """Split text into fixed-size chunks with overlap"""
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            chunks.append({
                'id': f'fixed_{chunk_id}',
                'text': chunk_text,
                'method': 'fixed_size',
                'start': start,
                'end': end
            })
            
            start = end - overlap
            chunk_id += 1
        
        return chunks
    
    def recursive_chunking(self, text: str, max_chunk_size: int = 1000) -> List[Dict]:
        """Split by paragraphs first, then sentences if needed"""
        chunks = []
        chunk_id = 0
        
        # Split by paragraphs
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(para) <= max_chunk_size:
                chunks.append({
                    'id': f'recursive_{chunk_id}',
                    'text': para,
                    'method': 'recursive',
                })
                chunk_id += 1
            else:
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= max_chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append({
                                'id': f'recursive_{chunk_id}',
                                'text': current_chunk.strip(),
                                'method': 'recursive',
                            })
                            chunk_id += 1
                        current_chunk = sentence + " "
                
                if current_chunk:
                    chunks.append({
                        'id': f'recursive_{chunk_id}',
                        'text': current_chunk.strip(),
                        'method': 'recursive',
                    })
                    chunk_id += 1
        
        return chunks
    
    def semantic_chunking(self, text: str, similarity_threshold: float = 0.7, 
                         max_chunk_size: int = 1000) -> List[Dict]:
        """Split text based on semantic similarity between sentences"""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return [{
                'id': 'semantic_0',
                'text': text,
                'method': 'semantic'
            }]
        
        # Get embeddings for all sentences
        embeddings = self.model.encode(sentences)
        
        chunks = []
        chunk_id = 0
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        
        for i in range(1, len(sentences)):
            # Calculate similarity with current chunk
            similarity = np.dot(current_embedding, embeddings[i]) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(embeddings[i])
            )
            
            current_text = ' '.join(current_chunk)
            
            # Check if we should start a new chunk
            if similarity < similarity_threshold or len(current_text) + len(sentences[i]) > max_chunk_size:
                chunks.append({
                    'id': f'semantic_{chunk_id}',
                    'text': current_text,
                    'method': 'semantic'
                })
                chunk_id += 1
                current_chunk = [sentences[i]]
                current_embedding = embeddings[i]
            else:
                current_chunk.append(sentences[i])
                # Update embedding to be average
                current_embedding = np.mean([current_embedding, embeddings[i]], axis=0)
        
        # Add last chunk
        if current_chunk:
            chunks.append({
                'id': f'semantic_{chunk_id}',
                'text': ' '.join(current_chunk),
                'method': 'semantic'
            })
        
        return chunks