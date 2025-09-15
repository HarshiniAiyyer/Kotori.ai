import uvicorn
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from kotori_graph import build_kotori_graph

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
if not hf_token:
    raise EnvironmentError("❌ Hugging Face token is missing! Please add HUGGINGFACE_API_TOKEN to your .env file.")

# Initialize FastAPI
app = FastAPI(title="Kotori.ai API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the LangGraph pipeline with error handling
graph = None
try:
    graph = build_kotori_graph()
    print("✅ Kotori.ai loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load Kotori.ai: {str(e)}")
    raise

# Define request model
class ChatRequest(BaseModel):
    input: str

# Chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest = Body(...)):
    global graph
    if graph is None:
        raise HTTPException(status_code=500, detail="Kotori.ai graph not initialized.")
    try:
        # Initialize state with required fields
        initial_state = {
            "input": request.input.strip(), 
            "response": "", 
            "agent": "", 
            "intent": ""
        }
        
        # Invoke the graph
        result = graph.invoke(initial_state)
        response = result.get("response", "⚠️ No response generated.")
        agent_used = result.get("agent", "unknown")
        
        # Validate response
        if not response or response.strip() == "":
            response = "I'm sorry, I couldn't generate a proper response. Please try rephrasing your question."
            
        return {
            "response": response,
            "agent": agent_used
        }
    except Exception as e:
        print(f"❌ API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get chat history endpoint (placeholder - implement actual history retrieval)
@app.get("/chat/history")
async def get_chat_history():
    # This is a placeholder. In a real implementation, you would retrieve
    # chat history from a database or other storage.
    return {
        "history": []
    }

# Clear chat history endpoint (placeholder - implement actual history clearing)
@app.delete("/chat/history")
async def clear_chat_history():
    # This is a placeholder. In a real implementation, you would clear
    # chat history from a database or other storage.
    return {
        "message": "Chat history cleared successfully"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Run the API server
if __name__ == "__main__":
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)