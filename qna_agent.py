import warnings
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*", category=FutureWarning)

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.schema import Document
from memory_utils import retrieve_memory, save_memory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ───────────────────────
# 1. ENV + EMBEDDINGS + VECTORSTORE
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

# GROQ LLM - RELIABLE AND FAST
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=200  # Reduced for concise responses
)

# ───────────────────────
# 2. CONCISE PROMPT TEMPLATE
# ───────────────────────
PROMPT_TEMPLATE = """You are Kotori, a caring assistant who helps people understand Empty Nest Syndrome. 

Provide a clear, concise answer using ONLY simple sentences. Use no more than 3 bullet points with good spacing between them. End with an engaging follow-up question.

Format your response exactly like this:
• [First key point in 1 simple sentence]

• [Second key point in 1 simple sentence]

• [Third key point in 1 simple sentence]

[Ask a simple follow-up question to continue the conversation]

**Context:**
{context}

**Question:** {question}

**Answer:**"""

PROMPT = PromptTemplate(input_variables=["context", "question"], template=PROMPT_TEMPLATE)
qna_chain = PROMPT | llm

# ───────────────────────
# 3. QnA Agent Node
# ───────────────────────
def qna_node(state: dict) -> dict:
    query = state.get("input", "")
    print(f"🔍 QnA processing: {query}")
    
    # Retrieve chunks from vectorstore
    try:
        relevant_chunks = vectorstore.similarity_search_with_score(query, k=3)  # Reduced for focus
        retrieved_texts = [doc.page_content for doc, _ in relevant_chunks]
        print(f"✅ Retrieved {len(retrieved_texts)} chunks from vectorstore")
    except Exception as e:
        print(f"⚠️ Vectorstore search error: {e}")
        retrieved_texts = []

    # Retrieve memory using utility
    try:
        past_texts = retrieve_memory(query, k=2)  # Reduced for focus
        print(f"✅ Retrieved {len(past_texts)} memories")
    except Exception as e:
        print(f"⚠️ Memory retrieval error: {e}")
        past_texts = []

    context = "\n\n---\n\n".join(retrieved_texts + past_texts)
    print(f"📝 Context length: {len(context)} characters")
    
    if not context.strip():
        state["response"] = """• Empty Nest Syndrome refers to feelings of sadness when children leave home.

• It's a normal part of parenting.

• These feelings are temporary.

What specific aspect of Empty Nest Syndrome would you like to know more about?"""
        state["agent"] = "qna"
        return state

    # LLM INVOCATION WITH GROQ
    print(f"🚀 About to call GROQ with context length: {len(context)}")
    try:
        limited_context = context[:4000]  # Increased context for more comprehensive responses
        print(f"🔄 Calling GROQ with limited context: {len(limited_context)} chars")
        
        result = qna_chain.invoke({
            "context": limited_context, 
            "question": query
        })
        
        print(f"✅ GROQ raw result type: {type(result)}")
        
        if hasattr(result, 'content'):
            response = result.content.strip()
        else:
            response = str(result).strip()
            
        print(f"✅ Final response: {response[:100]}...")
        
        # Validate response quality and format
        if not response or len(response) < 20:
            # Create a more query-specific fallback based on keywords in the query
            query_lower = query.lower()
            
            if "symptom" in query_lower or "sign" in query_lower:
                response = """• Empty Nest Syndrome causes feelings of sadness when children leave home.

• Parents may have trouble sleeping or feel less hungry.

• Some parents worry about their children or their own identity.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""
            elif "cause" in query_lower or "why" in query_lower:
                response = """• Empty Nest Syndrome happens when children leave home and parents' roles change.

• Parents may feel a void from fewer daily responsibilities.

• Your identity as a parent may feel challenged.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""
            elif "cope" in query_lower or "deal" in query_lower or "manage" in query_lower:
                response = """• Try reconnecting with activities you enjoy.

• Build new routines and keep in touch with your children.

• Talk to friends or a counselor for support.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""
            else:
                response = """• Empty Nest Syndrome is the sadness parents feel when children leave home.

• It's a normal feeling many parents experience.

• These feelings will pass with time.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""
        
        # Ensure proper format if missing
        if "•" not in response:
            # Make the formatting more query-specific
            query_lower = query.lower()
            follow_up = "What would you like to do next? Do you want to know more about empty nest? Or do you want to tell me how you are feeling today? Or shall I suggest activities to help you cope with this?"
                
            response = f"• {response}\n\n{follow_up}"
        
    except Exception as e:
        print(f"❌ GROQ error details: {e}")
        
        # Create a more query-specific error fallback based on keywords in the query
        query_lower = query.lower()
        
        if "symptom" in query_lower or "sign" in query_lower:
            response = """• Empty Nest Syndrome often causes feelings of sadness and loss.

• You might notice changes in your sleep or appetite.

• Many parents worry about their children or their own identity.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""
        elif "cause" in query_lower or "why" in query_lower:
            response = """• Empty Nest Syndrome happens when children leave home.

• Your daily routine changes without children at home.

• Many parents question their purpose during this time.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""
        elif "cope" in query_lower or "deal" in query_lower or "manage" in query_lower:
            response = """• Try new hobbies or return to old interests you enjoy.

• Keep in touch with your children while respecting their independence.

• Connect with other parents in similar situations.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""
        elif "calm" in query_lower or "remed" in query_lower or "help" in query_lower:
            response = """• Try mindfulness or meditation to manage your emotions.

• Regular exercise can reduce stress and improve your mood.

• Create new routines and set personal goals.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""
        else:
            response = """• Empty Nest Syndrome refers to feelings of sadness when children leave home.

• It's a natural part of parenting.

• These feelings are temporary.

What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?"""

    # Clean response
    response = response.replace("**Answer:**", "").strip()

    # Save memory using utility
    try:
        save_memory(query, response, memory_type="qna")
        print(f"✅ Saved to memory")
    except Exception as e:
        print(f"⚠️ Memory save error: {e}")

    # Update state
    state["response"] = response
    state["agent"] = "qna"
    return state