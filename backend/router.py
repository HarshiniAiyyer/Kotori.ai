import warnings
warnings.filterwarnings("ignore")

import os
import logging
from pathlib import Path
from typing import List
from difflib import SequenceMatcher
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document

# Load API tokens
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise EnvironmentError("❌ GROQ_API_KEY is missing in .env")

# Use Groq for fast, reliable routing
router_llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",  # Fast and accurate for classification
    temperature=0.1,  # Low temperature for consistent classification
    max_tokens=20   # Very short response needed for routing
)

def router_node(query: str) -> str:
    """Classifies user intent into qna, emotional, or suggestion using Groq"""
    
    query_lower = query.lower()
    greeting_keywords = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    # Check if query is a short greeting or contains only greeting words
    if any(query_lower.strip() == keyword for keyword in greeting_keywords) or \
       any(query_lower.strip().startswith(keyword + " ") for keyword in greeting_keywords) or \
       any(query_lower.strip().endswith(" " + keyword) for keyword in greeting_keywords) or \
       "hi kotori" in query_lower or "hello kotori" in query_lower or "hey kotori" in query_lower:
        print(f"✅ Routing '{query}' \u2192 welcome (greeting detected)")
        return "welcome"
        


    # Enhanced routing prompt with clear examples
    routing_prompt = f"""You are a classifier. Classify this user input into exactly ONE category: qna, emotional, or suggestion.

Examples:
- "What is Empty Nest Syndrome?" → qna
- "How do I cope with ENS?" → qna  
- "Tell me about empty nest syndrome" → qna
- "I feel sad today" → emotional
- "I'm lonely and depressed" → emotional
- "I miss my children" → emotional
- "Can you suggest activities?" → suggestion
- "Give me ways to feel better" → suggestion
- "What should I do now?" → suggestion

User input: "{query}"

Respond with only one word: qna, emotional, or suggestion"""

    try:
        print(f"🔀 Routing query: '{query[:50]}...' ")
        
        # Call Groq for classification
        result = router_llm.invoke(routing_prompt)
        
        # Extract response from ChatGroq
        if hasattr(result, 'content'):
            response = result.content.strip().lower()
        else:
            response = str(result).strip().lower()
        
        print(f"🤖 Groq raw response: '{response}'")
        
        # Extract the classification from response
        if "emotional" in response:
            intent = "emotional"
        elif "suggestion" in response:
            intent = "suggestion"
        elif "qna" in response:
            intent = "qna"
        else:
            # Enhanced fallback logic based on keywords
            query_lower = query.lower()
            
            # Emotional keywords (expanded)
            emotional_keywords = [
                "feel", "feeling", "sad", "lonely", "depressed", "upset", "cry", "crying", 
                "miss", "missing", "empty", "hurt", "hurting", "alone", "abandoned", 
                "lost", "grief", "mourn", "devastated", "heartbroken", "anxious", 
                "worried", "scared", "afraid", "overwhelmed", "helpless"
            ]
            
            # Suggestion keywords (expanded)
            suggestion_keywords = [
                "suggest", "suggestion", "recommend", "recommendation", "help me", 
                "what can i do", "what should i do", "how to", "ways to", "tips", 
                "advice", "ideas", "activities", "hobbies", "cope", "coping", 
                "deal with", "handle", "manage", "overcome"
            ]
            
            # QnA keywords (expanded)
            qna_keywords = [
                "what is", "what are", "explain", "define", "tell me about", 
                "how does", "why", "when", "where", "who", "definition", 
                "meaning", "understand", "learn", "know about"
            ]
            
            # Check emotional first (highest priority for support)
            if any(keyword in query_lower for keyword in emotional_keywords):
                intent = "emotional"
            # Then check for suggestions
            elif any(keyword in query_lower for keyword in suggestion_keywords):
                intent = "suggestion"
            # Then check for QnA
            elif any(keyword in query_lower for keyword in qna_keywords):
                intent = "qna"
            else:
                # Smart default based on query structure
                if "?" in query and any(word in query_lower for word in ["what", "how", "why", "when", "where"]):
                    intent = "qna"
                else:
                    intent = "qna"  # Safe default
        
        print(f"✅ Routing '{query}' → {intent}")
        return intent
    except Exception as e:
        print(f"❌ Groq routing error: {e}")
        
        # Robust keyword-based fallback
        query_lower = query.lower()
        
        # Prioritize emotional support
        if any(word in query_lower for word in ["feel", "sad", "lonely", "depressed", "upset", "miss", "hurt", "alone"]):
            fallback_intent = "emotional"
        elif any(word in query_lower for word in ["suggest", "recommend", "help me", "ways to", "tips", "advice", "what can i do"]):
            fallback_intent = "suggestion"
        else:
            fallback_intent = "qna"
        
        print(f"🔄 Using fallback routing: {fallback_intent}")
        return fallback_intent

# Test function for debugging
def test_router():
    """Test the router with various inputs"""
    test_cases = [
        ("What is Empty Nest Syndrome?", "qna"),
        ("I feel so sad and lonely", "emotional"),
        ("Can you suggest some activities?", "suggestion"),
        ("How do I cope with my children leaving?", "suggestion"),
        ("I miss my kids so much", "emotional"),
        ("Tell me about ENS symptoms", "qna")
    ]
    
    print("🧪 Testing Groq Router...")
    for query, expected in test_cases:
        try:
            result = router_node(query)
            status = "✅" if result == expected else "⚠️"
            print(f"{status} '{query}' → {result} (expected: {expected})")
        except Exception as e:
            print(f"❌ Test failed for '{query}': {e}")

__all__ = ["router_node", "test_router"]

if __name__ == "__main__":
    test_router()

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
DATA_DIR = os.getenv("DATA_DIR", "./data")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

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
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

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
        chroma = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=get_embeddings())
        
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
        
        chroma.persist()
        logger.info("✅ Successfully persisted Chroma database")
        
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