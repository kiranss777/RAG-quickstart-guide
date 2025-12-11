import streamlit as st
import PyPDF2
from dotenv import load_dotenv
import os
from chunking import DocumentChunker
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from llm_chain import DeepSeekLLM

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="RAG PDF Chat", page_icon="📚", layout="wide")

# Get API keys from environment variables only
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'embedder' not in st.session_state:
    st.session_state.embedder = None
if 'current_document' not in st.session_state:
    st.session_state.current_document = None

# Title
st.title("📚 RAG PDF Chat Application")
st.caption("Using Pinecone Index: ads7390ragapp | Model: all-MiniLM-L12-v2 | Dimensions: 384")

# Check for API keys
if not PINECONE_API_KEY or not DEEPSEEK_API_KEY:
    st.error("⚠️ Missing API keys! Please set PINECONE_API_KEY and DEEPSEEK_API_KEY in your .env file")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("📄 Document Upload")
    
    # PDF Upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    # Chunking method
    chunking_method = st.selectbox(
        "Select Chunking Method",
        ["Semantic Chunking", "Fixed Size Chunking", "Recursive Chunking"]
    )
    
    # Process button
    if st.button("Process PDF", type="primary"):
        if not uploaded_file:
            st.error("Please upload a PDF file")
        else:
            with st.spinner("Processing PDF..."):
                try:
                    # Initialize vector store if not already done
                    if st.session_state.vector_store is None:
                        st.session_state.vector_store = VectorStore(PINECONE_API_KEY)
                    
                    # Clear the index before processing new document
                    st.warning("🗑️ Clearing previous data from index...")
                    st.session_state.vector_store.delete_all()
                    
                    # Reset chat history and LLM
                    st.session_state.chat_history = []
                    if st.session_state.llm:
                        st.session_state.llm.clear_history()
                    
                    # Read PDF
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    
                    st.info(f"📄 Extracted {len(text)} characters from PDF")
                    
                    # Initialize components
                    chunker = DocumentChunker()
                    if st.session_state.embedder is None:
                        st.session_state.embedder = EmbeddingGenerator(PINECONE_API_KEY)
                    if st.session_state.llm is None:
                        st.session_state.llm = DeepSeekLLM(DEEPSEEK_API_KEY)
                    
                    # Chunk the text
                    st.info("✂️ Chunking document...")
                    if chunking_method == "Semantic Chunking":
                        chunks = chunker.semantic_chunking(text)
                    elif chunking_method == "Fixed Size Chunking":
                        chunks = chunker.fixed_size_chunking(text)
                    else:
                        chunks = chunker.recursive_chunking(text)
                    
                    st.session_state.chunks = chunks
                    st.success(f"✅ Created {len(chunks)} chunks")
                    
                    # Generate embeddings
                    st.info("🔢 Generating embeddings...")
                    texts = [chunk['text'] for chunk in chunks]
                    embeddings = st.session_state.embedder.generate_embeddings(texts)
                    st.success(f"✅ Generated {len(embeddings)} embeddings (384 dimensions)")
                    
                    # Upsert to Pinecone
                    st.info("☁️ Uploading to Pinecone index 'ads7390ragapp'...")
                    st.session_state.vector_store.upsert_chunks(chunks, embeddings)
                    
                    st.session_state.processed = True
                    st.session_state.current_document = uploaded_file.name
                    st.success(f"🎉 All done! Ready to chat with your document!")
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Document info
    if st.session_state.processed:
        st.divider()
        st.header("Document Info")
        st.metric("Current Document", st.session_state.current_document)
        st.metric("Total Chunks", len(st.session_state.chunks))
        st.metric("Chunking Method", st.session_state.chunks[0]['method'])
        
        if st.button("Clear Document & Index"):
            with st.spinner("Clearing..."):
                st.session_state.processed = False
                st.session_state.chunks = []
                st.session_state.chat_history = []
                st.session_state.current_document = None
                if st.session_state.llm:
                    st.session_state.llm.clear_history()
                if st.session_state.vector_store:
                    st.session_state.vector_store.delete_all()
            st.success("✅ Cleared!")
            st.rerun()

# Main chat interface
if st.session_state.processed:
    st.header("💬 Chat with your Document")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "chunks" in message:
                with st.expander("📎 Source Chunks Used"):
                    for chunk in message["chunks"]:
                        st.markdown(f"**Chunk ID:** `{chunk['id']}` | **Similarity Score:** `{chunk['score']:.4f}`")
                        st.caption(f"Method: {chunk['method']}")
                        st.text(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])
                        st.divider()
    
    # Chat input
    if query := st.chat_input("Ask a question about your document..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.write(query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching and generating response..."):
                # Get query embedding
                query_embedding = st.session_state.embedder.generate_embedding_single(query)
                
                # Retrieve relevant chunks
                retrieved_chunks = st.session_state.vector_store.query(query_embedding, top_k=3)
                
                # Generate response
                response = st.session_state.llm.generate_response(query, retrieved_chunks)
                
                st.write(response)
                
                # Show source chunks
                with st.expander("📎 Source Chunks Used"):
                    for chunk in retrieved_chunks:
                        st.markdown(f"**Chunk ID:** `{chunk['id']}` | **Similarity Score:** `{chunk['score']:.4f}`")
                        st.caption(f"Method: {chunk['method']}")
                        st.text(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])
                        st.divider()
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "chunks": retrieved_chunks
                })

else:
    st.info("👈 Upload a PDF and configure settings in the sidebar to get started!")
    
    # Show example
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚀 How to use:
        1. Make sure your `.env` file has API keys
        2. Upload a PDF document
        3. Select a chunking method
        4. Click "Process PDF"
        5. Start asking questions!
        
        **Note:** Processing a new PDF will clear the previous document from the index.
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Chunking Methods:
        - **Semantic**: Groups by meaning (Recommended)
        - **Fixed Size**: Equal-sized chunks with overlap
        - **Recursive**: Splits by paragraphs → sentences
        """)
    
    st.divider()
    st.info("🔧 **Your Pinecone Index:** ads7390ragapp | **Embedding Model:** all-MiniLM-L12-v2 | **Dimensions:** 384")