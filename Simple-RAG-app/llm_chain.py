from typing import List, Dict
from openai import OpenAI

class DeepSeekLLM:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = "deepseek-chat"
        self.chat_history = []
    
    def generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate response using DeepSeek with retrieved context"""
        
        # Prepare context from chunks
        context = "\n\n".join([
            f"[Chunk {chunk['id']}]: {chunk['text']}"
            for chunk in context_chunks
        ])
        
        # Create prompt
        system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
Always reference the chunk IDs you used to answer the question. 
If the context doesn't contain relevant information, say so clearly."""
        
        user_prompt = f"""Context:
{context}

Question: {query}

Please answer the question based on the context above. Mention which chunk IDs you used."""
        
        # Add to chat history
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Include recent chat history (last 3 exchanges)
        for msg in self.chat_history[-6:]:
            messages.append(msg)
        
        messages.append({"role": "user", "content": user_prompt})
        
        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        
        # Update chat history
        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": answer})
        
        return answer
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []