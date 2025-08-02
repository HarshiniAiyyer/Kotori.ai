from langchain.schema import Document
from langchain_chroma import Chroma  # Updated import
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
import os
import hashlib

# ─────────────────────────────
# 1. ChromaDB setup (shared)
# ─────────────────────────────
# Use relative path for deployment compatibility
CHROMA_DIR = Path(os.getenv("CHROMA_DB_PATH", Path(__file__).parent / "chroma"))

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Create directory if it doesn't exist
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embedding_model
)

# ─────────────────────────────
# 2. Save conversation to memory
# ─────────────────────────────
def save_memory(query: str, response: str, memory_type: str = "qna") -> None:
    """
    Saves a user-assistant interaction to Chroma vectorstore.
    """
    memory_doc = Document(
        page_content=f"User: {query}\nAssistant: {response}",
        metadata={
            "source": "chat_memory",
            "type": memory_type,
            "id": f"conv_{hashlib.sha256(query.encode()).hexdigest()}"
        }
    )
    try:
        vectorstore.add_documents([memory_doc], ids=[memory_doc.metadata["id"]])
    except Exception as e:
        print(f"⚠️ Could not save to memory: {e}")

# ─────────────────────────────
# 3. Retrieve past memory chunks
# ─────────────────────────────
def retrieve_memory(query, k=5):
    """
    Retrieves k most relevant past memories related to the query,
    with improved prioritization of recent and relevant memories.
    """
    try:
        # Try direct search first with increased k for better coverage
        results = vectorstore.similarity_search_with_score(query, k=k+2)  # Get extra results to filter
        memory_docs = []
        
        for doc, score in results:
            # Only include documents from chat_memory
            if doc.metadata.get("source") == "chat_memory":
                # Store the document with its score and metadata for better filtering
                memory_docs.append((doc, score))
        
        # Sort by relevance (score) first
        memory_docs.sort(key=lambda x: x[1])
        
        # Extract the memory texts from the sorted documents
        memory_texts = []
        memory_types_count = {"qna": 0, "emotional": 0, "suggestion": 0}
        
        # First pass: prioritize diverse memory types
        for doc, score in memory_docs:
            memory_type = doc.metadata.get("type", "unknown")
            
            # Ensure we have a balanced mix of memory types
            if memory_type in memory_types_count and memory_types_count[memory_type] < 2:
                memory_texts.append(doc.page_content)
                memory_types_count[memory_type] += 1
                
        # Second pass: add remaining memories up to k
        remaining_slots = k - len(memory_texts)
        if remaining_slots > 0:
            for doc, score in memory_docs:
                if doc.page_content not in memory_texts and len(memory_texts) < k:
                    memory_texts.append(doc.page_content)
                
        print(f"🧠 Found {len(memory_texts)} relevant memories with improved diversity")
        return memory_texts
        
    except Exception as e:
        print(f"⚠️ Could not retrieve memory: {e}")
        return []