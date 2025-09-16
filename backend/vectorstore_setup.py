import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Use an absolute path for the Chroma database
# Default to backend/chroma if not set
CHROMA_DIR = Path(os.getenv("CHROMA_DB_PATH", Path(__file__).parent / "chroma"))
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

print(f"Chroma DB Path: {CHROMA_DIR.resolve()}")

# Initialize embedding model
_embedding_model = None
_vectorstore = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="paraphrase-MiniLM-L3-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embedding_model

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR.resolve()),
            embedding_function=get_embedding_model()
        )
    return _vectorstore