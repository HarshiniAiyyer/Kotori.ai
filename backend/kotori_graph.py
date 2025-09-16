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
from vectorstore_setup import get_vectorstore
from loader import load_pdfs, split_docs, deduplicate_chunks, save_to_chroma, load_environment

MAX_CHAT_HISTORY_LENGTH = 5

# Define the state for the LangGraph
class KotoriState(TypedDict):
    input: str
    response: str
    agent: Literal["qna", "emotional", "suggestion", "welcome", "router"]
    chat_history: Annotated[List[str], {"max_length": MAX_CHAT_HISTORY_LENGTH}]

# Console-only logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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
