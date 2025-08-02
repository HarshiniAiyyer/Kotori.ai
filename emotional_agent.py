import warnings
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*", category=FutureWarning)

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.schema import Document
from memory_utils import save_memory
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise EnvironmentError("❌ GROQ_API_KEY missing.")

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Load Chroma vectorstore
# Use relative path for deployment compatibility
CHROMA_DIR = Path(os.getenv("CHROMA_DB_PATH", Path(__file__).parent / "chroma"))
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

vectorstore = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embedding_model)

# GROQ LLM for emotional support
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.4,  # Slightly higher for more natural emotional responses
    max_tokens=200  # Reduced for concise responses
)

# Concise Prompt Template for Emotional Support
EMOTION_PROMPT_TEMPLATE = """You are Kotori, a compassionate assistant helping with Empty Nest Syndrome.

Provide warm, supportive response using ONLY simple sentences. Use no more than 3 bullet points with good spacing between them. End with options for what to do next.

Format your response exactly like this:
• [Validate their feelings in 1 simple sentence]

• [Offer comfort or reassurance in 1 simple sentence]

• [Provide gentle encouragement in 1 simple sentence]

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?

**Context:**
{context}

**User's Message:** {question}

**Supportive Response:**"""

EMOTION_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=EMOTION_PROMPT_TEMPLATE
)

emotional_chain = EMOTION_PROMPT | llm

# LangGraph-compatible node function
def emotional_checkin_node(state: dict) -> dict:
    query = state.get("input", "")
    print(f"💝 Emotional support processing: {query}")
    
    # Retrieve context from Chroma
    try:
        docs = vectorstore.similarity_search_with_score(query, k=2)  # Reduced for focus
        context = "\n\n---\n\n".join([doc.page_content for doc, _ in docs])
        print(f"✅ Retrieved context for emotional support: {len(context)} chars")
    except Exception as e:
        print(f"⚠️ Vectorstore search error: {e}")
        context = ""

    # If no context, provide general emotional support context
    if not context.strip():
        context = "Empty Nest Syndrome is a common experience where parents feel sadness, loneliness, or loss of purpose when their children leave home. These feelings are completely normal and temporary."

    # Invoke chain with GROQ
    try:
        print(f"🚀 Calling GROQ for emotional support...")
        
        result = emotional_chain.invoke({
            "context": context[:3000],  # Increased context for better emotional support
            "question": query
        })
        
        # Extract response from ChatGroq
        if hasattr(result, 'content'):
            response = result.content.strip()
        else:
            response = str(result).strip()
            
        print(f"✅ Generated emotional response: {response[:100]}...")
        
        # Validate response and format
        if not response or len(response) < 20:
            # Create a more query-specific fallback based on keywords in the query
            query_lower = query.lower()
            
            if "sad" in query_lower or "depress" in query_lower or "down" in query_lower or "blue" in query_lower:
                response = """• It's normal to feel sad when your children leave home.

• Many parents feel down during this time.

• Talk to a professional if your sadness feels too heavy.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""
            elif "lonely" in query_lower or "alone" in query_lower or "empty" in query_lower:
                response = """• Feeling empty is normal when your home changes.

• Many parents struggle with this big change in their life.

• This is a chance to rediscover yourself.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""
            elif "purpose" in query_lower or "meaning" in query_lower or "identity" in query_lower:
                response = """• It's normal to question your purpose after being a parent for so long.

• This time can help you grow in new ways.

• Give yourself time to adjust to this change.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""
            else:
                response = """• Your feelings are valid.

• Many parents find this time hard.

• Be kind to yourself during this change.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""
        
        # Ensure proper format if missing
        if "•" not in response:
            # Make the formatting more query-specific
            query_lower = query.lower()
            intro = "I understand you're going through a difficult time."
            follow_up = "What would you like to do next? Do you want to know more about empty nest? Or do you want to tell me how you are feeling today? Or shall I suggest activities to help you cope with this?"
            
            response = f"• {intro} {response}\n\n{follow_up}"
            
    except Exception as e:
        print(f"❌ GROQ error in emotional agent: {e}")
        
        # Create a more query-specific error fallback based on keywords in the query
        query_lower = query.lower()
        
        if "sad" in query_lower or "depress" in query_lower or "down" in query_lower or "blue" in query_lower:
            response = """• The sadness you're feeling is a natural response to this significant life change.
• Many parents experience similar feelings of loss and grief when children leave home.
• These emotions, while difficult, often become less intense as you adjust to your new normal.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""
        elif "lonely" in query_lower or "alone" in query_lower or "empty" in query_lower:
            response = """• The emptiness of your home can be one of the most challenging aspects of this transition.
• This feeling of loneliness is shared by many parents adjusting to children's departure.
• Creating new routines and connections can gradually help fill the space that feels empty now.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""
        elif "purpose" in query_lower or "meaning" in query_lower or "identity" in query_lower or "lost" in query_lower:
            response = """• Many parents feel a sense of lost purpose when their primary caregiving role changes.
• This transition is an opportunity to rediscover aspects of yourself beyond parenting.
• Finding new meaning often comes through exploring interests and connections you couldn't fully pursue before.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""
        else:
            response = """• I hear you, and your feelings are completely valid and normal.
• Empty Nest Syndrome is challenging, but these emotions will ease with time.
• You're not alone in this - many parents successfully navigate this transition.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""

    # Clean response
    response = response.replace("**Supportive Response:**", "").strip()

    # Save memory using utility
    try:
        save_memory(query, response, memory_type="emotional")
        print(f"✅ Saved emotional interaction to memory")
    except Exception as e:
        print(f"⚠️ Memory save error: {e}")

    # Return new state
    state["response"] = response
    state["agent"] = "emotional"
    return state