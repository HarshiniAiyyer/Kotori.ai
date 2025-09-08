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
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Initialize Chroma vectorstore
vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR.resolve()),
    embedding_function=embedding_model
)

def get_vectorstore():
    return vectorstore

def get_embedding_model():
    return embedding_model