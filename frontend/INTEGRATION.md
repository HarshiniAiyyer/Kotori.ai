# Integrating the Modern Frontend with Kotori.ai Backend

This document provides instructions on how to integrate the new React-based frontend with the existing Kotori.ai Streamlit backend.

## Overview

The modern frontend is built with React and communicates with the backend via API calls. There are two approaches to integration:

1. **API Integration**: Modify the existing Streamlit app to expose API endpoints
2. **Standalone Deployment**: Run both applications separately

## Option 1: API Integration

### Step 1: Add FastAPI to the Streamlit Backend

Create a new file `api.py` in the root directory:

```python
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from kotori_graph import build_kotori_graph
import uvicorn

# Initialize FastAPI
app = FastAPI(title="Kotori.ai API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the LangGraph pipeline
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
        raise HTTPException(status_code=500, detail=str(e))

# Run the API server
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
```

### Step 2: Install Required Dependencies

```bash
pip install fastapi uvicorn
```

### Step 3: Run the API Server

```bash
python api.py
```

## Option 2: Standalone Deployment

### Step 1: Configure the React App

Create a `.env` file in the `modern-frontend` directory:

```
REACT_APP_API_URL=http://localhost:8000
```

### Step 2: Run Both Applications

**Terminal 1 (Backend):**
```bash
python app2.py  # Run the Streamlit app
```

**Terminal 2 (Frontend):**
```bash
cd modern-frontend
npm install
npm start
```

## Production Deployment Considerations

1. **Build the React App**:
   ```bash
   cd modern-frontend
   npm run build
   ```

2. **Serve the Static Files**:
   You can serve the built React app using a static file server like Nginx or Apache.

3. **API Security**:
   - Implement proper authentication for API endpoints
   - Set specific CORS origins
   - Consider rate limiting

4. **Environment Variables**:
   - Use environment variables for configuration
   - Keep API keys and secrets secure

## Conclusion

This modern frontend provides a more accessible and user-friendly interface for users aged 48+ while maintaining all the functionality of the original Streamlit application. The design focuses on readability, clear navigation, and a calm, professional aesthetic suitable for the target audience.