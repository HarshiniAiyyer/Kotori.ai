import os
import logging
from pathlib import Path
from typing import List
from difflib import SequenceMatcher
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from vectorstore_setup import get_vectorstore, get_embedding_model

# Console-only logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
SIMILARITY_THRESHOLD = 0.95

# Paths
# Correctly set paths relative to the project root
BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BACKEND_DIR / "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT_DIR / "chroma"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Load env and token
def load_environment() -> bool:
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
    if hf_token:
        os.environ["HUGGINGFACE_API_TOKEN"] = hf_token
        logger.info("✅ Hugging Face token loaded")
        return True
    else:
        logger.warning("⚠️ HUGGINGFACE_API_TOKEN not found in .env")
        return False

# Embedding function
def get_embeddings():
    return get_embedding_model()

# Load PDFs
def load_pdfs() -> List[Document]:
    try:
        logger.info(f"🔍 Looking for PDFs in: {DATA_DIR}")
        # Check what files are in the directory
        pdf_files = list(Path(DATA_DIR).glob("*.pdf"))
        logger.info(f"📂 Found {len(pdf_files)} PDF files in directory")
        if pdf_files:
            logger.info("📑 PDF files found:")
            for pdf in pdf_files:
                logger.info(f"   - {pdf.name} (Size: {pdf.stat().st_size} bytes)")
        
        loader = PyPDFDirectoryLoader(str(DATA_DIR))
        documents = loader.load()
        logger.info(f"📄 Successfully loaded {len(documents)} documents")
        if documents:
            logger.info(f"📝 First document metadata: {documents[0].metadata}")
        return documents
    except Exception as e:
        logger.error(f"❌ Failed to load PDFs: {e}", exc_info=True)
        return []

# Split docs
def split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    logger.info(f"🔪 Split into {len(chunks)} chunks")
    return chunks

# Deduplicate
def is_similar(a: str, b: str, threshold: float = SIMILARITY_THRESHOLD) -> bool:
    return SequenceMatcher(None, a, b).ratio() > threshold

def deduplicate_chunks(chunks: List[Document]) -> List[Document]:
    seen, unique = [], []
    for chunk in tqdm(chunks, desc="🧹 Deduplicating"):
        text = chunk.page_content.strip()
        if not any(is_similar(text, s) for s in seen):
            seen.append(text)
            unique.append(chunk)
    logger.info(f"✨ {len(unique)} unique chunks kept from {len(chunks)}")
    return unique

# Assign IDs
def assign_chunk_ids(chunks: List[Document]) -> List[Document]:
    last_page_id = None
    chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        page_id = f"{source}:{page}"
        chunk_index = chunk_index + 1 if page_id == last_page_id else 0
        chunk.metadata["id"] = f"{page_id}:{chunk_index}"
        last_page_id = page_id
    return chunks

# Save to Chroma
def clean_text(text: str) -> str:
    """Clean and validate text before embedding."""
    if not text or not isinstance(text, str):
        return ""
    # Remove any non-printable characters and extra whitespace
    import re
    text = re.sub(r'[^\x20-\x7E\n\r\t]', ' ', text)
    text = ' '.join(text.split())
    return text.strip()

def save_to_chroma(chunks: List[Document]):
    try:
        # Initialize Chroma
        chroma = get_vectorstore()
        
        # Assign IDs and clean text content
        chunks = assign_chunk_ids(chunks)
        
        # Get existing IDs
        try:
            existing = chroma.get(include=[])
            existing_ids = set(existing["ids"])
        except Exception as e:
            logger.warning(f"Could not fetch existing IDs: {e}")
            existing_ids = set()

        # Process and validate chunks
        valid_chunks = []
        for chunk in chunks:
            try:
                # Clean the text content
                chunk.page_content = clean_text(chunk.page_content)
                if not chunk.page_content:
                    logger.warning(f"Skipping empty chunk: {chunk.metadata.get('id', 'unknown')}")
                    continue
                if chunk.metadata["id"] not in existing_ids:
                    valid_chunks.append(chunk)
            except Exception as e:
                logger.error(f"Error processing chunk {chunk.metadata.get('id', 'unknown')}: {e}")
                continue

        if not valid_chunks:
            logger.info("✅ No new valid chunks to add")
            return

        logger.info(f"💾 Adding {len(valid_chunks)} new chunks")
        
        # Process in smaller batches
        batch_size = 50  # Reduced batch size for better stability
        for i in tqdm(range(0, len(valid_chunks), batch_size), desc="🔗 Saving"):
            batch = valid_chunks[i:i + batch_size]
            try:
                chroma.add_documents(
                    documents=batch,
                    ids=[c.metadata["id"] for c in batch]
                )
            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                # Try adding documents one by one to identify the problematic one
                for doc in batch:
                    try:
                        chroma.add_documents(
                            documents=[doc],
                            ids=[doc.metadata["id"]]
                        )
                    except Exception as single_doc_error:
                        logger.error(f"Failed to add document {doc.metadata['id']}: {single_doc_error}")
                        continue
        logger.info("✅ Chroma database updated")
        
    except Exception as e:
        logger.error(f"❌ Error in save_to_chroma: {e}", exc_info=True)
        raise

# Main
def main():
    if not load_environment():
        logger.warning("🚫 Environment setup incomplete, continuing anyway")

    docs = load_pdfs()
    if not docs:
        logger.error("📭 No documents found. Exiting.")
        return

    chunks = split_docs(docs)
    clean_chunks = deduplicate_chunks(chunks)
    save_to_chroma(clean_chunks)
    logger.info("🏁 Done!")

if __name__ == "__main__":
    main()