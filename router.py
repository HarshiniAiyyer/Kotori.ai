import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load API tokens
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Use Groq if available, otherwise provide fallback
if groq_api_key:
    # Use Groq for fast, reliable routing
    router_llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",  # Fast and accurate for classification
        temperature=0.1,  # Low temperature for consistent classification
        max_tokens=20   # Very short response needed for routing
    )
else:
    # Fallback: Use a simple rule-based router for deployment
    print("⚠️ GROQ_API_KEY not found, using fallback routing")
    router_llm = None  # Will be handled in the routing function

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
        
    # Check for structured follow-up responses
    if "know more about empty nest" in query_lower or "tell me about empty nest" in query_lower:
        print(f"✅ Routing '{query}' \u2192 qna (follow-up selection)")
        return "qna"
    elif "tell me how you are feeling" in query_lower or "how i am feeling" in query_lower or "how i feel" in query_lower:
        print(f"✅ Routing '{query}' \u2192 emotional (follow-up selection)")
        return "emotional"
    elif "suggest activities" in query_lower or "activities to help" in query_lower or "help me cope" in query_lower:
        print(f"✅ Routing '{query}' \u2192 suggestion (follow-up selection)")
        return "suggestion"

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
        
        # Use Groq if available, otherwise use fallback logic
        if router_llm is not None:
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
                intent = None  # Will use fallback below
        else:
            print("⚠️ Using fallback routing (no Groq API key)")
            intent = None  # Use fallback logic
        
        # Fallback logic when Groq is unavailable or unclear
        if intent is None:
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