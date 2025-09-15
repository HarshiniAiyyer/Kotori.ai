# Kotori.ai - Your Companion for Empty Nest Syndrome

Kotori.ai is your companion designed to support parents experiencing Empty Nest Syndrome. It provides information, emotional support, and practical suggestions to help navigate this significant life transition.

## 🚀 Project Overview

Kotori.ai is an innovative AI assistant designed to provide comprehensive support for parents navigating Empty Nest Syndrome. It offers a blend of informational resources, empathetic emotional support, and practical suggestions to help individuals gracefully transition through this significant life stage. The project is built with a focus on accessibility and user-friendliness, particularly for users aged 48 and above, ensuring a seamless and supportive experience.

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



### 💻 Running the Application Locally

For development, you can use the provided batch script:

```bash
# Run both the API server and React frontend
run_dev.bat
```

This will start:
- The API server at http://localhost:8000
- The React frontend at http://localhost:3000

### 📦 Building for Production

```bash
# Build the React app for production
build_frontend.bat
```

The production-ready files will be available in the `frontend/build` directory.

## Project Structure

### Backend Components

- `backend/vectorstore_setup.py` - Configuration for the vector store and embedding model. Currently uses `all-MiniLM-L6-v2` for optimized memory usage.
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

## ☁️ Deployment Options

Refer to `deploy-checklist.md` for detailed deployment instructions for both backend and frontend.

## 📄 License

This project is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited.

## Acknowledgments

- Built with LangGraph, Groq, and Hugging Face technologies
- Designed with accessibility and usability in mind for users aged 48+