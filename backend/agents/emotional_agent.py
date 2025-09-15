import warnings
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*", category=FutureWarning)

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.schema import Document
from memory_utils import save_memory
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from vectorstore_setup import get_vectorstore

# Load .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise EnvironmentError("❌ GROQ_API_KEY missing.")

# Get the vectorstore instance
vectorstore = get_vectorstore()

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

End with a single, engaging follow-up question that encourages further conversation or exploration of their feelings, without offering predefined options. The question should be open-ended and empathetic.

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
            
            response = "I hear you, and your feelings are completely valid and normal. Empty Nest Syndrome is challenging, but these emotions will ease with time. You're not alone in this - many parents successfully navigate this transition. How are you feeling about all of this right now?"
        
        # Ensure proper format if missing
        if "•" not in response:
            response = f"• {response}"
            
    except Exception as e:
        print(f"❌ GROQ error in emotional agent: {e}")
        
        # Create a more query-specific error fallback based on keywords in the query
        query_lower = query.lower()
        
        response = "I hear you, and your feelings are completely valid and normal. Empty Nest Syndrome is challenging, but these emotions will ease with time. You're not alone in this - many parents successfully navigate this transition. How are you feeling about all of this right now?"

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
