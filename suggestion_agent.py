import warnings
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*", category=FutureWarning)

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.schema import Document
from memory_utils import save_memory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ───────────────────────
# 1. ENV + EMBEDDINGS + DB + LLM
# ───────────────────────
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise EnvironmentError("❌ GROQ_API_KEY is missing.")

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Use relative path for deployment compatibility
CHROMA_DIR = Path(os.getenv("CHROMA_DB_PATH", Path(__file__).parent / "chroma"))
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

vectorstore = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embedding_model)

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

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?

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
            
            if "calm" in query_lower or "relax" in query_lower or "stress" in query_lower or "anxious" in query_lower or "anxiety" in query_lower:
                response = """• Try a short daily meditation to manage worry.

• Practice simple breathing: in for 4 counts, hold for 4, out for 6.

• Create a calm bedtime routine for better sleep.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""
            elif "hobby" in query_lower or "activit" in query_lower or "interest" in query_lower:
                response = """• Try creative activities like painting, writing, or music.

• Physical hobbies like walking, dancing, or yoga can boost your mood.

• Volunteering helps you connect with others.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""
            elif "social" in query_lower or "connect" in query_lower or "friend" in query_lower or "lonely" in query_lower:
                response = """• Volunteer for causes you care about to meet new people.

• Join clubs or classes that match your interests.

• Reconnect with old friends you haven't seen in a while.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""
            else:
                response = """• Try reconnecting with old friends or joining community groups.

• Explore new hobbies or return to old interests you enjoy.

• Create a routine that includes time for yourself.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""
        
        # Ensure proper format if missing
        if "•" not in response:
            # Make the formatting more query-specific
            query_lower = query.lower()
            follow_up = "What would you like to do next? Do you want to know more about empty nest? Or do you want to tell me how you are feeling today? Or shall I suggest activities to help you cope with this?"
                
            response = f"• {response}\n\n{follow_up}"
            
    except Exception as e:
        print(f"❌ GROQ error in suggestion agent: {e}")
        
        # Create a more query-specific error fallback based on keywords in the query
        query_lower = query.lower()
        
        if "calm" in query_lower or "relax" in query_lower or "stress" in query_lower:
            response = """• Practice deep breathing exercises: inhale for 4 counts, hold for 4, exhale for 6 counts.
• Create a daily relaxation ritual like a warm bath with essential oils or a quiet reading session.
• Try a guided meditation app to help manage stress and improve sleep quality.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""
        elif "hobby" in query_lower or "activit" in query_lower or "interest" in query_lower:
            response = """• Consider exploring new hobbies or interests that you may have put on hold during active parenting.
• Try rotating through different activities weekly until you find ones that truly engage you.
• Look for local classes or workshops to learn new skills in a social environment.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""
        elif "social" in query_lower or "connect" in query_lower or "friend" in query_lower or "lonely" in query_lower:
            response = """• Join community groups or classes aligned with your interests to meet like-minded people.
• Consider volunteering for causes you care about as a meaningful way to connect with others.
• Reach out to old friends or neighbors for coffee dates or walks to rebuild your social circle.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""
        else:
            response = """• Consider exploring new hobbies or interests that you may have put on hold during active parenting.
• Reconnect with friends and family, or join community groups to build new social connections.
• Focus on personal wellness through exercise, meditation, or other self-care activities.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""

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