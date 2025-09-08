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

## 🏗️ Architecture & Tech Stack

### 📖 RAG Literature Summary

Here is a summary of the literature used for the RAG (Retrieval-Augmented Generation) system, categorized by article type:

1.  **1-Empty-Nest-PDF.pdf** \
    *Type:* Magazine article (narrative feature) \
    *Summary:* A personal essay exploring the emotional impact of children leaving home, highlighting parental grief, identity loss, and the transition to a new life phase. It includes expert insights and cultural commentary.

2.  **2-.pdf** \ 
    *Type:* Research paper (academic study) \
    *Summary:* Examines the relationship between empty nest syndrome and life satisfaction among Indian middle-aged adults. Finds significant gender differences in ENS and a negative correlation with life satisfaction.

3.  **3-.pdf** \
    *Type:* Research paper (peer-reviewed journal article) \
    *Summary:* A mixed-methods study exploring parental gender and cultural differences in empty nest syndrome among four ethnic groups in Canada. Finds that Indo/East Indian parents report higher levels of ENS.

4.  **4-mtnest-indian form.pdf** \
    *Type:* Research paper (scale development and validation) \
    *Summary:* Describes the development and validation of the Empty Nest Syndrome Scale—Indian Form (ENS-IF), a tool to measure ENS among Indian parents, including psychometric properties and cultural relevance.

5.  **5-Review_Article_2.pdf** \
    *Type:* Review article (critical clinical review)
    *Summary:* Critically evaluates empty nest syndrome from clinical, cultural, and gender perspectives. Discusses its validity, neurobiological factors, and relevance in the digital age.

6.  **6-www-cadabamshospitals-com-empty-nest-syndrome-helpful-tips-to-overcome-.pdf** \
    *Type:* Internet article (health advice blog) \
    *Summary:* Offers practical advice on coping with empty nest syndrome, including stages, symptoms, and self-help strategies. Promotes professional support and highlights benefits of the empty nest phase.

7.  **7-www.cbetterhealth.vic.gov.pdf** \
    *Type:* Government health fact sheet \
    *Summary:* Provides an overview of empty nest syndrome, its emotional impact, coping strategies, and planning advice. Targets parents, especially mothers, and includes resource links.

8.  **8-.pdf** \
    *Type:* University web article (promotional educational content) \
    *Summary:* Promotes a course on thriving during the empty nest phase. Emphasizes reframing the experience as an opportunity for growth, resilience, and personal development.

9.  **9-health_clevelandclinic.pdf** \
    *Type:* Health clinic article (expert advice) \
    *Summary:* A psychologist shares 10 tips for coping with empty nest syndrome, including communication, self-care, and knowing when to seek help. Includes causes, symptoms, and reassurance.

10. **10-www_unitegroup.pdf** \ 
    *Type:* Corporate blog post (research highlight) \
    *Summary:* Shares survey findings on parental emotions when children leave for university. Discusses the impact of ENS and the role of higher education institutions in supporting parents.

11. **11-merthyrtydfil_fosterwales.gov.pdf** \
    *Type:* Government blog post (promotional storytelling) \
    *Summary:* Uses personal fostering stories to address empty nest syndrome. Encourages fostering as a way to fill the emotional void and contribute to the community.

### 🧠 Backend Components

The backend is built with Python and leverages the following key technologies:

- **FastAPI**: A modern, fast (high-performance) web framework for building APIs with Python 3.9+ based on standard Python type hints.
- **LangChain**: Used for orchestrating complex AI workflows, including conversational memory and agentic behavior.
- **LangGraph**: A library for building robust and stateful multi-actor applications with LLMs, enabling the creation of sophisticated conversational agents.
- **LLM**: Groq (llama-3.3-70b-versatile)
- **Hugging Face Embeddings**: Used for generating vector embeddings from text, crucial for semantic search and context retrieval. Specifically, the `BAAI/bge-small-en-v1.5` model is employed for its efficiency and accuracy.
- **Hugging Face**: Utilized for various NLP tasks and potentially for specific models or embeddings.
- **Pinecone**: A vector database used for efficient storage and retrieval of embeddings, enabling semantic search and context management.

Key backend modules include:
- `backend/kotori_graph.py`: Defines the core LangGraph application flow.
- `backend/router.py`: Handles intent classification to direct user queries to appropriate agents.
- `backend/agents/`: Contains specialized AI agents (`qna_agent.py`, `emotional_agent.py`, `suggestion_agent.py`) for different types of support.
- `backend/memory_utils.py`: Manages conversation history and context for persistent interactions.
- `backend/api.py`: Implements the FastAPI server, exposing endpoints for the frontend to interact with the AI assistant.

### 🌐 Frontend Components

The frontend is a modern web application built with React, focusing on accessibility and an intuitive user experience, especially for an older demographic.

- **React**: A JavaScript library for building user interfaces, providing a component-based architecture for modular and reusable UI elements.
- **Tailwind CSS**: A utility-first CSS framework used for rapidly building custom designs directly in your HTML.
- **PostCSS**: A tool for transforming CSS with JavaScript plugins, used here with Autoprefixer for vendor prefixing.
- **Axios**: A promise-based HTTP client for making API requests to the backend.
- **React Router DOM**: For declarative routing within the single-page application.

Key frontend directories:
- `frontend/`: The root of the React application.
  - `src/pages/`: Contains the main views and pages of the application.
  - `src/components/`: Houses reusable UI components used across different pages.
  - `src/services/`: Manages API communication logic, abstracting backend interactions.

## ☁️ Deployment Options

Refer to `deploy-checklist.md` for detailed deployment instructions for both backend and frontend.

## 📄 License

This project is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited.

## 🙏 Acknowledgments


- Built with LangGraph, Groq, and Hugging Face technologies.
