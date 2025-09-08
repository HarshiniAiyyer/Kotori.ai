# Kotori.ai - Empty Nest Syndrome Support Assistant

Kotori.ai is an AI assistant designed to support parents experiencing Empty Nest Syndrome. It provides information, emotional support, and practical suggestions to help navigate this significant life transition.

## Project Overview

This project consists of two main components:

1. **Original Streamlit Application** - The core AI assistant built with Streamlit, LangGraph, and Groq LLM.
2. **Modern React Frontend** - A new, accessible frontend designed specifically for users aged 48+ with a focus on readability and ease of use.

## Getting Started

### Prerequisites

- Python 3.9+ (for the backend)
- Node.js 14+ (for the modern frontend)
- Groq API key
- Hugging Face API token

### Environment Setup

1. Create a `.env` file in the root directory with the following variables:

```
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
```

### Running the Original Streamlit App

```bash
# Activate the virtual environment
cd go\Scripts
activate
cd ../..

# Run the Streamlit app
python app2.py
```

The Streamlit app will be available at http://localhost:8501

### Running the Modern Frontend with API Backend

For development, you can use the provided batch script:

```bash
# Run both the API server and React frontend
run_dev.bat
```

This will start:
- The API server at http://localhost:8000
- The React frontend at http://localhost:3000

### Building the Modern Frontend for Production

```bash
# Build the React app for production
build_frontend.bat
```

The production-ready files will be available in the `frontend/build` directory.

## Project Structure

### Backend Components

- `backend/kotori_graph.py` - Main LangGraph definition
- `backend/router.py` - Intent classification for user queries
- `backend/agents/qna_agent.py` - Information-providing agent
- `backend/agents/emotional_agent.py` - Emotional support agent
- `backend/agents/suggestion_agent.py` - Practical suggestions agent
- `backend/memory_utils.py` - Conversation memory management
- `backend/api.py` - FastAPI server for the modern frontend

### Frontend Components

- `frontend/` - React application designed for accessibility
  - `src/pages/` - Main application pages
  - `src/components/` - Reusable UI components
  - `src/services/` - API communication services

## Deployment Options

Refer to `deploy-checklist.md` for detailed deployment instructions for both backend and frontend.

## License

This project is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited.

## Acknowledgments

- Built with LangGraph, Groq, and Hugging Face technologies
- Designed with accessibility and usability in mind for users aged 48+