from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from typing_extensions import Annotated
from langgraph.graph import StateGraph, END, START

# Import router + agents
from router import router_node
from agents.qna_agent import qna_node as qna_agent_node
from agents.emotional_agent import emotional_checkin_node as emotional_agent_node
from agents.suggestion_agent import suggestion_node as suggestion_agent_node
from agents.welcome_agent import welcome_agent_node
import os
import logging
from pathlib import Path
from typing import List
from difflib import SequenceMatcher
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from vectorstore_setup import get_vectorstore

# Define the state for the LangGraph
class KotoriState(TypedDict):
    input: str
    response: str
    agent: Literal["qna", "emotional", "suggestion", "welcome", "router"]
    chat_history: List[str]

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

# ─────────────────────────────
# 2. Agent Nodes (Fixed to properly handle state)
# ─────────────────────────────
def qna_node(state: KotoriState) -> KotoriState:
    try:
        result = qna_agent_node(state)
        return result
    except Exception as e:
        print(f"❌ Error in QnA node: {e}")
        state["response"] = f"Sorry, I encountered an error while processing your question: {str(e)}"
        state["agent"] = "qna"
        return state

def emotional_node(state: KotoriState) -> KotoriState:
    try:
        result = emotional_agent_node(state)
        return result
    except Exception as e:
        print(f"❌ Error in emotional node: {e}")
        state["response"] = f"I understand you're reaching out for emotional support. I'm here to help, but I encountered a technical issue: {str(e)}"
        state["agent"] = "emotional"
        return state

def suggestion_node(state: KotoriState) -> KotoriState:
    try:
        result = suggestion_agent_node(state)
        return result
    except Exception as e:
        print(f"❌ Error in suggestion node: {e}")
        state["response"] = f"I'd love to provide some suggestions, but I encountered an error: {str(e)}"
        state["agent"] = "suggestion"
        return state

def welcome_node(state: KotoriState) -> KotoriState:
    try:
        result = welcome_agent_node(state["input"])
        state["response"] = result
        state["agent"] = "welcome"
        return state
    except Exception as e:
        print(f"❌ Error in welcome node: {e}")
        state["response"] = f"Hello! I'm Kotori, your companion for navigating Empty Nest Syndrome. What would you like to do next? Do you want to know more about empty nest? Or do you want to tell me how you are feeling today? Or shall I suggest activities to help you cope with this?"
        state["agent"] = "welcome"
        return state

# ─────────────────────────────
# 3. Router function (FIXED)
# ─────────────────────────────
def route_user_input(state: KotoriState) -> Literal["qna", "emotional", "suggestion", "welcome"]:
    """Uses router_node to classify intent"""
    try:
        # Extract just the input string and pass to router
        intent = router_node(state["input"])
        
        # Update state with the determined intent
        state["intent"] = intent
        
        print(f"🔀 Routing to: {intent}")
        return intent
    except Exception as e:
        print(f"❌ Router error: {e}")
        # Default to qna if routing fails
        state["intent"] = "qna"
        return "qna"

# ─────────────────────────────
# 4. Build LangGraph (IMPROVED)
# ─────────────────────────────
def build_kotori_graph():
    workflow = StateGraph(KotoriState)

    # Add nodes
    workflow.add_node("qna", qna_node)
    workflow.add_node("emotional", emotional_node)
    workflow.add_node("suggestion", suggestion_node)
    workflow.add_node("welcome", welcome_node)

    # Add conditional routing from START
    workflow.add_conditional_edges(
        START,
        route_user_input,
        {
            "qna": "qna",
            "emotional": "emotional",
            "suggestion": "suggestion",
            "welcome": "welcome"
        }
    )

    # All nodes end at END
    workflow.add_edge("qna", END)
    workflow.add_edge("emotional", END)
    workflow.add_edge("suggestion", END)
    workflow.add_edge("welcome", END)

    return workflow.compile()

# ─────────────────────────────
# 5. Test function for debugging
# ─────────────────────────────
def test_graph():
    """Test function to debug the graph"""
    try:
        graph = build_kotori_graph()
        
        test_queries = [
            "What is empty nest syndrome?",
            "I feel so lonely since my kids left home",
            "Can you suggest some activities for me?"
        ]
        
        for query in test_queries:
            print(f"\n🧪 Testing: '{query}'")
            try:
                state = {"input": query, "response": "", "agent": "", "intent": ""}
                result = graph.invoke(state)
                print(f"✅ Response: {result['response'][:100]}...")
                print(f"📍 Agent used: {result['agent']}")
            except Exception as e:
                print(f"❌ Test failed: {e}")
                
    except Exception as e:
        print(f"❌ Graph build failed: {e}")

# ─────────────────────────────
# CLI (optional)
# ─────────────────────────────
if __name__ == "__main__":
    # Test the graph first
    print("🧪 Testing graph...")
    test_graph()
    
    print("\n🌐 Kotori is live.")
    # graph = build_kotori_graph() # This line is commented out to prevent rebuilding the graph on every run
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        try:
            state = {"input": user_input, "response": "", "agent": "", "intent": ""}
            final_state = graph.invoke(state)
            print(f"\nKotori: {final_state['response']}\n")
        except Exception as e:
            print(f"❌ Error: {e}")
