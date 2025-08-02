import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load API tokens
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise EnvironmentError("❌ GROQ_API_KEY is missing in .env")

# Initialize Groq LLM for welcome messages
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.7, # A bit higher for more varied greetings
    max_tokens=50
)

# Prompt template for welcome messages
WELCOME_PROMPT_TEMPLATE = """You are Kotori, a friendly and caring companion for navigating Empty Nest Syndrome. Your purpose is to greet the user warmly and offer clear options for how you can help them today.

Respond with a warm, friendly greeting using only simple sentences. Keep it concise and inviting. After your greeting, offer clear options for what they might want to do next.

Example:
User: Hi Kotori!
Kotori: Hello! I'm Kotori, your companion for navigating Empty Nest Syndrome. What would you like to do next? Do you want to know more about empty nest? Do you want to tell me how you are feeling today? Shall I suggest activities to help you cope with this?

User's Message: {query}

Your Greeting:"""

WELCOME_PROMPT = PromptTemplate(input_variables=["query"], template=WELCOME_PROMPT_TEMPLATE)
welcome_chain = WELCOME_PROMPT | llm

def welcome_agent_node(query: str) -> str:
    """Handles welcome messages and provides a friendly greeting."""
    try:
        print(f"👋 Invoking welcome agent for query: '{query}'")
        response = welcome_chain.invoke({"query": query})
        print(f"👋 Welcome agent raw response: {response}")
        if hasattr(response, 'content'):
            content = response.content.strip()
            print(f"👋 Welcome agent content: {content}")
            if content:
                return content
            else:
                print("👋 Empty content from welcome agent, using fallback greeting")
                return "Hello! I'm Kotori, your companion for navigating Empty Nest Syndrome. What would you like to do next? Do you want to know more about empty nest? Or do you want to tell me how you are feeling today? Or shall I suggest activities to help you cope with this?"
        else:
            content = str(response).strip()
            print(f"👋 Welcome agent string response: {content}")
            if content:
                return content
            else:
                print("👋 Empty string response from welcome agent, using fallback greeting")
                return "Hello! I'm Kotori, your companion for navigating Empty Nest Syndrome. What would you like to do next? Do you want to know more about empty nest? Or do you want to tell me how you are feeling today? Or shall I suggest activities to help you cope with this?"
    except Exception as e:
        print(f"❌ Error in welcome agent: {e}")
        return "Hello! I'm Kotori, your companion for navigating Empty Nest Syndrome. What would you like to do next? Do you want to know more about empty nest? Or do you want to tell me how you are feeling today? Or shall I suggest activities to help you cope with this?" # Fallback greeting

__all__ = ["welcome_agent_node"]