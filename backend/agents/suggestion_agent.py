import warnings
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*", category=FutureWarning)

import os
import logging
from pathlib import Path
from typing import List
from difflib import SequenceMatcher
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from vectorstore_setup import get_vectorstore
from langchain.schema.document import Document

# ───────────────────────
# 1. ENV + EMBEDDINGS + DB + LLM
# ───────────────────────
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise EnvironmentError("❌ GROQ_API_KEY is missing.")

vectorstore = get_vectorstore()

# GROQ LLM for suggestions
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.5,  # Higher temperature for more creative suggestions
    max_tokens=200  # Reduced for concise responses
)

# ───────────────────────
# 2. CONCISE SUGGESTION PROMPT
# ───────────────────────
SUGGESTION_TEMPLATE = """You are Kotori, helping people navigate life after Empty Nest Syndrome.

Provide practical suggestions using ONLY simple sentences. Use no more than 3 bullet points with good spacing between them. End with options for what to do next.

Format your response exactly like this:
• [First practical suggestion in 1 simple sentence]

• [Second practical suggestion in 1 simple sentence]

• [Third practical suggestion in 1 simple sentence]

End with a single, engaging follow-up question that encourages further exploration of practical steps or personal interests, without offering predefined options. The question should be open-ended and actionable.

**Context:**
{context}

**User's Request:** {question}

**Helpful Suggestions:**"""

SUGGESTION_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=SUGGESTION_TEMPLATE
)

suggestion_chain = SUGGESTION_PROMPT | llm

# ───────────────────────
# 3. Concise Suggestion Agent
# ───────────────────────
def suggestion_node(state: dict) -> dict:
    query = state.get("input", "")
    print(f"💡 Suggestion processing: {query}")

    # Retrieve suggestions-related content from memory or documents
    try:
        suggestion_chunks = vectorstore.similarity_search_with_score(query, k=3)  # Reduced for focus
        context_chunks = [doc.page_content for doc, _ in suggestion_chunks]
        print(f"✅ Retrieved {len(context_chunks)} suggestion-related chunks")
    except Exception as e:
        print(f"⚠️ Vectorstore search error: {e}")
        context_chunks = []

    context = "\n\n---\n\n".join(context_chunks)
    
    # If no specific context, provide general Empty Nest context
    if not context.strip():
        context = "Empty Nest Syndrome often involves feelings of loneliness and loss of purpose after children leave home. Common coping strategies include exploring new hobbies, maintaining social connections, focusing on self-care, and discovering new life purposes."
        print("📝 Using general Empty Nest context")

    print(f"📝 Context length: {len(context)} characters")

    try:
        print(f"🚀 Calling GROQ for suggestions...")
        
        result = suggestion_chain.invoke({
            "context": context[:3500],  # Increased context for better suggestions
            "question": query
        })
        
        # Extract response from ChatGroq
        if hasattr(result, 'content'):
            response = result.content.strip()
        else:
            response = str(result).strip()
            
        print(f"✅ Generated suggestions: {response[:100]}...")
        
        # Validate response and format
        if not response or len(response) < 30:
            # Create a more query-specific fallback based on keywords in the query
            query_lower = query.lower()
            
            response = "It sounds like you're looking for ways to navigate this new phase. Many find it helpful to explore new activities or reconnect with their community. What kind of suggestions are you hoping for today?"
        
        # Ensure proper format if missing
        if "•" not in response:
            response = f"• {response}"
            
    except Exception as e:
        print(f"❌ GROQ error in suggestion agent: {e}")
        
        # Create a more query-specific error fallback based on keywords in the query
        query_lower = query.lower()
        
        response = "It sounds like you're looking for ways to navigate this new phase. Many find it helpful to explore new activities or reconnect with their community. What kind of suggestions are you hoping for today?"

    # Clean response
    response = response.replace("**Helpful Suggestions:**", "").strip()

    # Save memory using utility
    try:
        save_memory(query, response, memory_type="suggestion")
        print(f"✅ Saved suggestions to memory")
    except Exception as e:
        print(f"⚠️ Memory save error: {e}")

    # Update state
    state["response"] = response
    state["agent"] = "suggestion"
    return state

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
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "./chroma"))
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