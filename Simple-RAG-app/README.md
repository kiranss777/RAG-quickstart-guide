# RAG PDF Chat

A simple Retrieval-Augmented Generation (RAG) application that ingests PDF documents, chunks them, generates embeddings, stores vectors in Pinecone, and lets you chat with the document using a DeepSeek-compatible LLM.

## Project structure

- `app.py` - Streamlit app UI and orchestration
- `chunking.py` - Document chunking utilities using `sentence-transformers`
- `embeddings.py` - Embedding generation using `sentence-transformers` (384-dim models)
- `vector_store.py` - Pinecone index management and query
- `llm_chain.py` - LLM wrapper (uses `openai.OpenAI` pointing to DeepSeek)

## Requirements

Install dependencies from `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

Note: `sentence-transformers` will pull in `transformers` and `torch`.

## Environment variables

Create a `.env` file in the project root with the following keys:

- `PINECONE_API_KEY` — Your Pinecone API key
- `DEEPSEEK_API_KEY` — Your DeepSeek/OpenAI-compatible API key

Example `.env`:

```
PINECONE_API_KEY=your_pinecone_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
```

## Running the app

Start the Streamlit app:

```powershell
streamlit run app.py
```

Open the Streamlit URL shown in the terminal, upload a PDF in the sidebar, choose a chunking method, and click "Process PDF". After processing, use the chat interface to ask questions about your document.
