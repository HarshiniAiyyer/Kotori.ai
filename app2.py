import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from kotori_graph import build_kotori_graph

# ─────────────────────────────────────────
# 1. Setup
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Kotori.ai - Empty Nest Support", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
if not hf_token:
    st.error("❌ Hugging Face token is missing! Please add HUGGINGFACE_API_TOKEN to your .env file.")
    st.stop()

# Load the LangGraph pipeline with error handling
try:
    graph = build_kotori_graph()
    st.success("✅ Kotori.ai loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load Kotori.ai: {str(e)}")
    st.error("Please check your configuration and try again.")
    st.stop()

# ─────────────────────────────────────────
# 2. Enhanced Custom Styling
# ─────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main app background - soft lavender like logo background */
    .stApp {
        background: linear-gradient(135deg, #e8f0fe 0%, #f0f4ff 100%);
        color: #2c3e50;
    }
    
    /* Main container styling */
    .main > div {
        padding: 2rem 1rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Title styling */
    .main-title {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Section header styling - bigger and more readable */
    .section-header {
        font-size: 28px !important;
        color: #2c5aa0 !important;
        font-weight: 600 !important;
        margin: 30px 0 20px 0 !important;
        padding-bottom: 12px;
        border-bottom: 3px solid rgba(173, 216, 230, 0.5);
        text-align: center;
    }
    
    /* Quick start section with extra spacing */
    .quick-start-header {
        font-size: 28px !important;
        color: #2c5aa0 !important;
        font-weight: 600 !important;
        margin: 40px 0 25px 0 !important;
        padding-bottom: 12px;
        border-bottom: 3px solid rgba(173, 216, 230, 0.5);
        text-align: center;
    }
    
    /* Quick action container with spacing */
    .quick-action-container {
        background: rgba(248, 251, 255, 0.8);
        padding: 25px;
        border-radius: 12px;
        margin: 25px 0;
        border: 1px solid rgba(173, 216, 230, 0.3);
        box-shadow: 0 2px 8px rgba(173, 216, 230, 0.1);
    }
    
    .sub-title {
        font-size: 1.5rem !important;
        color: #34495e !important;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Input field styling */
    div[data-testid="stTextInput"] {
        margin: 2rem 0;
    }
    
    div[data-testid="stTextInput"] input {
        font-size: 18px !important;
        padding: 15px !important;
        border-radius: 8px !important;
        border: 2px solid #b8d4ea !important;
        background-color: #ffffff !important;
        color: #2c3e50 !important;
        box-shadow: 0 2px 8px rgba(173, 216, 230, 0.2);
        height: 56px !important; /* Fixed height for alignment */
        box-sizing: border-box !important;
    }
    
    div[data-testid="stTextInput"] input:focus {
        border-color: #5a7ba8 !important;
        box-shadow: 0 0 0 3px rgba(90, 123, 168, 0.2) !important;
        outline: none;
    }
    
    /* Response container styling - soft teal like logo bird */
    .response-container {
        background: linear-gradient(135deg, #e8f4f8 0%, #d4edda 100%);
        color: #2c3e50;
        padding: 25px;
        border-radius: 12px;
        margin: 25px 0;
        box-shadow: 0 4px 15px rgba(173, 216, 230, 0.2);
        border-left: 4px solid #5a7ba8;
    }
    
    .response-container p {
        font-size: 19px !important;
        line-height: 1.8 !important;
        margin: 0 !important;
        color: #2c3e50 !important;
        font-weight: 400;
    }
    
    .agent-info {
        font-size: 16px !important;
        color: #6b7280 !important;
        margin-top: 15px !important;
        font-style: italic;
        text-align: right;
    }
    
    /* Sidebar styling - ensure #B8CDEB color */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: #B8CDEB !important;
    }
    
    /* Alternative sidebar selectors for different Streamlit versions */
    .css-1d391kg > div, [data-testid="stSidebar"] > div {
        background: #B8CDEB !important;
    }
    
    /* Sidebar text color adjustment for light background */
    .css-1d391kg .stMarkdown, [data-testid="stSidebar"] .stMarkdown {
        color: #2c3e50 !important;
    }
    
    .css-1d391kg h2, [data-testid="stSidebar"] h2 {
        color: #2c5aa0 !important;
    }
    
    .css-1d391kg p, [data-testid="stSidebar"] p {
        color: #4a5568 !important;
        line-height: 1.6 !important;
    }
    
    /* Button styling - using logo blue colors */
    div.stButton > button {
        background: linear-gradient(135deg, #5a7ba8 0%, #4a6590 100%);
        color: white !important;
        font-weight: 500;
        font-size: 16px !important;
        border-radius: 8px;
        padding: 15px 24px;
        border: none;
        box-shadow: 0 3px 12px rgba(90, 123, 168, 0.25);
        transition: all 0.2s ease;
        height: 56px; /* Match input height */
    }
    
    div.stButton > button:hover {
        background: linear-gradient(135deg, #4a6590 0%, #3a5578 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(90, 123, 168, 0.35);
    }
    
    /* Primary button styling */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2c5aa0 0%, #1e4080 100%);
        height: 56px; /* Match input height */
    }
    
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1e4080 0%, #163060 100%);
    }
    
    /* Expander styling - logo colors */
    .streamlit-expanderHeader {
        background-color: rgba(173, 216, 230, 0.2) !important;
        border-radius: 5px;
        color: #2c3e50 !important;
        font-size: 15px !important;
    }
    
    .streamlit-expanderContent {
        background-color: rgba(248, 251, 255, 0.8) !important;
        border-radius: 5px;
        color: #2c3e50 !important;
    }
    
    /* Error container styling - softer red */
    .error-container {
        background: linear-gradient(135deg, #fef2f2 0%, #fed7d7 100%);
        color: #742a2a;
        border-left: 4px solid #e53e3e;
        padding: 20px;
        margin: 15px 0;
        border-radius: 8px;
        box-shadow: 0 3px 12px rgba(229, 62, 62, 0.15);
    }
    
    .error-container h4 {
        color: #742a2a !important;
        margin: 0 0 10px 0 !important;
    }
    
    .error-container p, 
    .error-container li {
        color: #744210 !important;
        font-size: 16px !important;
    }
    
    /* Tips section styling - soft blue matching the theme */
    .tips-container {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        color: #2c3e50;
        padding: 30px;
        border-radius: 12px;
        margin: 30px 0;
        box-shadow: 0 4px 20px rgba(173, 216, 230, 0.2);
        border-left: 4px solid #5a7ba8;
        border: 1px solid rgba(173, 216, 230, 0.3);
    }
    
    .tips-title {
        font-size: 26px !important;
        font-weight: 600 !important;
        color: #2c5aa0 !important;
        margin-bottom: 20px !important;
        text-shadow: none;
    }
    
    .tips-content {
        font-size: 18px !important;
        line-height: 1.7 !important;
        color: #2c3e50 !important;
        font-weight: 400;
    }
    
    .tips-content li {
        margin-bottom: 12px !important;
        color: #4a5568 !important;
    }
    
    .tips-content strong {
        color: #2c5aa0 !important;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #5a7ba8 !important;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: rgba(209, 250, 229, 0.8) !important;
        border: 1px solid #9ae6b4 !important;
        color: #276749 !important;
        font-size: 16px !important;
    }
    
    /* Footer styling */
    .footer-container {
        background: linear-gradient(135deg, #f8fbff 0%, #e8f4f8 100%);
        padding: 40px 20px 20px 20px;
        margin-top: 50px;
        border-top: 3px solid rgba(173, 216, 230, 0.4);
        border-radius: 15px 15px 0 0;
    }
    
    .footer-content {
        max-width: 800px;
        margin: 0 auto;
        text-align: center;
    }
    
    .social-icons {
        display: flex;
        justify-content: center;
        gap: 25px;
        margin: 25px 0;
        flex-wrap: wrap;
    }
    
    .social-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        text-decoration: none;
        transition: all 0.3s ease;
        font-size: 24px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    .social-icon:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .email-icon {
        background: linear-gradient(135deg, #5a7ba8 0%, #4a6590 100%);
        color: white;
    }
    
    .linkedin-icon {
        background: linear-gradient(135deg, #0077b5 0%, #005885 100%);
        color: white;
    }
    
    .github-icon {
        background: linear-gradient(135deg, #333 0%, #24292e 100%);
        color: white;
    }
    
    .footer-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #b8d4ea 50%, transparent 100%);
        margin: 30px 0 20px 0;
        border: none;
    }
    
    .team-credit {
        font-size: 18px;
        color: #2c5aa0;
        font-weight: 500;
        margin-bottom: 15px;
    }
    
    .footer-tagline {
        font-size: 16px;
        color: #6b7280;
        font-style: italic;
        margin-bottom: 10px;
    }
    
    .footer-copyright {
        font-size: 14px;
        color: #9ca3af;
        margin-top: 20px;
    }
    
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 1.5s ease-in-out infinite;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ─────────────────────────────────────────
# 3. Enhanced Page Header
# ─────────────────────────────────────────
st.markdown('''
<div style="text-align: center; padding: 2rem 0;">
    <h1 class="main-title">Hi, I am Kotori.</h1>
    <h2 class="sub-title">Your Compassionate Companion for Empty Nest Syndrome</h2>
    <p style="font-size: 1.2rem; color: #7f8c8d; margin-bottom: 2rem;">
        Feeling lonely without your kids? You might be experiencing Empty Nest Syndrome. Let Kotori help you navigate this transition.
    </p>
</div>
''', unsafe_allow_html=True)

# ─────────────────────────────────────────
# 4. Enhanced Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    # Use relative path for deployment compatibility
    image_path = Path(__file__).parent / "assets" / "images" / "image.png"
    if Path(image_path).exists():
        st.image(image_path, use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: rgba(173, 216, 230, 0.2); border-radius: 10px; margin-bottom: 20px; border: 1px solid rgba(173, 216, 230, 0.4);">
            <h2 style="color: #2c5aa0; margin: 0; font-size: 24px;">Kotori</h2>
            <p style="color: #5a7ba8; margin: 5px 0 0 0; font-weight: 500;">AI Companion</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## About Kotori")
    st.markdown("""
    **Kotori** is your AI companion designed specifically to help parents navigate Empty Nest Syndrome. 
    
    Whether you need:
    - **Information** about Empty Nest Syndrome
    - **Emotional support** during difficult moments  
    - **Practical suggestions** for moving forward
    
    Kotori is here to listen and help.
    """)

    st.markdown("## Recent Conversations")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if st.session_state["chat_history"]:
        for i, chat in enumerate(reversed(st.session_state["chat_history"][-5:])):  # Show last 5
            with st.expander(f"Conversation: {chat['query'][:40]}...", expanded=False):
                st.markdown(f"**Q:** {chat['query']}")
                st.markdown(f"**A:** {chat['response'][:200]}...")
    else:
        st.markdown("*No conversations yet. Start by asking a question!*")

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()

# ─────────────────────────────────────────
# 5. Enhanced Query Input Section
# ─────────────────────────────────────────
st.markdown('<div class="section-header">How can Kotori help you today?</div>', unsafe_allow_html=True)

# Initialize quick query if button was clicked
if "quick_query" not in st.session_state:
    st.session_state.quick_query = ""

# Create columns for better layout with proper alignment
col1, col2 = st.columns([4, 1], gap="small")

with col1:
    # Use the quick_query if available, otherwise empty
    default_value = st.session_state.quick_query if st.session_state.quick_query else ""
    query = st.text_input(
        "Ask your question:", 
        value=default_value,
        placeholder="Tell me about Empty Nest Syndrome, share your feelings, or ask for suggestions...", 
        key="query_input",
        label_visibility="collapsed"
    )

with col2:
    # Add some CSS to ensure button aligns with input
    st.markdown('<div style="margin-top: 0px;">', unsafe_allow_html=True)
    search_button = st.button("Ask Kotori", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# 6. Process Query
# ─────────────────────────────────────────
# Clear the quick_query after using it
if st.session_state.quick_query:
    st.session_state.quick_query = ""

if (query and query.strip()) or search_button:
    if query and query.strip():
        with st.spinner("Kotori is thinking..."):
            try:
                # Initialize state with required fields
                initial_state = {
                    "input": query.strip(), 
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
                
                # Save to session history (avoid duplicates)
                current_entry = {"query": query, "response": response}
                if (not st.session_state["chat_history"] or 
                    st.session_state["chat_history"][-1]["query"] != query):
                    st.session_state["chat_history"].append(current_entry)
                    
                # Keep only last 15 entries
                if len(st.session_state["chat_history"]) > 15:
                    st.session_state["chat_history"].pop(0)

                # Format and display response with readable styling
                formatted_response = (response
                                    .replace("•", "•")  # Keep simple bullet points
                                    .replace("\n\n", "<br><br>")
                                    .replace("\n", "<br>"))

                # Agent names without emojis
                agent_names = {
                    "qna": "Information Assistant",
                    "emotional": "Emotional Support", 
                    "suggestion": "Suggestion Assistant",
                    "welcome": "Welcome Assistant"
                }
                agent_name = agent_names.get(agent_used, "Assistant")

                st.markdown(
                    f"""
                    <div class="response-container">
                        <div style="display: flex; align-items: center; margin-bottom: 18px;">
                            <span style="font-size: 22px; font-weight: 500; color: #2c5aa0;">Kotori's Response</span>
                        </div>
                        <div style="font-size: 19px; line-height: 1.8; margin-bottom: 15px; color: #2c3e50;">
                            {formatted_response}
                        </div>
                        <div class="agent-info">
                            {agent_name} • Powered by Kotori.ai
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:
                # Enhanced error display
                error_msg = str(e)
                st.markdown(
                    f"""
                    <div class="error-container">
                        <h4>⚠️ Oops! Something went wrong</h4>
                        <p>Kotori encountered an issue while processing your question. Here are some things you can try:</p>
                        <ul>
                            <li><strong>Rephrase your question</strong> - Try asking in a different way</li>
                            <li><strong>Ask a simpler question</strong> - Break complex questions into parts</li>
                            <li><strong>Check your connection</strong> - Ensure you have internet access</li>
                        </ul>
                        <details style="margin-top: 15px;">
                            <summary style="cursor: pointer; font-weight: bold;">🔧 Technical Details (Click to expand)</summary>
                            <div style="background: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px; margin-top: 10px; font-family: monospace; font-size: 12px;">
                                {error_msg}
                            </div>
                        </details>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Log error for debugging
                print(f"❌ Streamlit app error: {e}")

# ─────────────────────────────────────────
# 7. Enhanced Tips Section
# ─────────────────────────────────────────
st.markdown("---")

st.markdown("""
<div class="tips-container">
    <div class="tips-title">Tips for Better Conversations with Kotori</div>
    <div class="tips-content">
        <ul style="list-style: none; padding-left: 0;">
            <li><strong>Ask specific questions</strong> about Empty Nest Syndrome symptoms, causes, or coping strategies</li>
            <li><strong>Share your feelings</strong> if you need emotional support - Kotori is here to listen and validate your experience</li>
            <li><strong>Request practical suggestions</strong> for activities, hobbies, or ways to reconnect with yourself</li>
            <li><strong>Be conversational</strong> - You can say things like "I feel lonely" or "What should I do now?"</li>
            <li><strong>Follow up</strong> - Ask follow-up questions to dive deeper into topics that interest you</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 8. Quick Action Buttons
# ─────────────────────────────────────────
st.markdown('<div class="quick-start-header">Quick Start Options</div>', unsafe_allow_html=True)

# Add container for better spacing
st.markdown('<div class="quick-action-container">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Learn About Empty Nest", use_container_width=True):
        st.session_state.quick_query = "What is Empty Nest Syndrome and what are its symptoms?"
        st.rerun()

with col2:
    if st.button("Get Emotional Support", use_container_width=True):
        st.session_state.quick_query = "I'm feeling lonely and sad since my children left home"
        st.rerun()

with col3:
    if st.button("Get Activity Suggestions", use_container_width=True):
        st.session_state.quick_query = "Can you suggest activities to help me cope with empty nest syndrome?"
        st.rerun()

# Close the quick action container
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# 9. Enhanced Footer
# ─────────────────────────────────────────
# Enhanced Footer with proper rendering
st.markdown("---")

# Inspirational message
st.markdown("""
<div class="footer-container">
    <div class="footer-content">
        <p style="font-size: 20px; margin-bottom: 20px; color: #2c5aa0; font-weight: 500; line-height: 1.6; text-align: center;">
            <strong>Remember:</strong> Empty Nest Syndrome is a natural part of parenting.<br>
            You're not alone in this journey. 🌸
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Social media icons section with proper spacing
st.markdown('<div style="text-align: center; margin: 25px 0; padding: 20px 0;">', unsafe_allow_html=True)

# Create columns for better icon spacing
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown('''
    <div style="text-align: center;">
        <a href="mailto:team@kotori.ai" title="Email Us" style="text-decoration: none;">
            <div style="display: inline-block; width: 55px; height: 55px; border-radius: 50%; background: linear-gradient(135deg, #5a7ba8 0%, #4a6590 100%); color: white; text-align: center; line-height: 55px; font-size: 24px; box-shadow: 0 4px 12px rgba(90, 123, 168, 0.3); transition: all 0.3s ease; cursor: pointer;">
                ✉️
            </div>
        </a>
        <div style="margin-top: 8px; font-size: 12px; color: #5a7ba8; font-weight: 500;">Email</div>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown('''
    <div style="text-align: center;">
        <a href="https://linkedin.com/company/kotori-ai" title="LinkedIn" target="_blank" style="text-decoration: none;">
            <div style="display: inline-block; width: 55px; height: 55px; border-radius: 50%; background: linear-gradient(135deg, #0077b5 0%, #005885 100%); color: white; text-align: center; line-height: 55px; font-size: 24px; box-shadow: 0 4px 12px rgba(0, 119, 181, 0.3); transition: all 0.3s ease; cursor: pointer;">
                💼
            </div>
        </a>
        <div style="margin-top: 8px; font-size: 12px; color: #0077b5; font-weight: 500;">LinkedIn</div>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown('''
    <div style="text-align: center;">
        <a href="https://github.com/kotori-ai" title="GitHub" target="_blank" style="text-decoration: none;">
            <div style="display: inline-block; width: 55px; height: 55px; border-radius: 50%; background: linear-gradient(135deg, #333 0%, #24292e 100%); color: white; text-align: center; line-height: 55px; font-size: 24px; box-shadow: 0 4px 12px rgba(51, 51, 51, 0.3); transition: all 0.3s ease; cursor: pointer;">
                🐱
            </div>
        </a>
        <div style="margin-top: 8px; font-size: 12px; color: #333; font-weight: 500;">GitHub</div>
    </div>
    ''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Clean Footer - Perfect HTML Rendering
st.markdown("---")

# Tagline - broken into individual elements
st.markdown('<div style="text-align: center; margin: 30px 0 15px 0;">', unsafe_allow_html=True)
st.markdown('<h3 style="color: #2c5aa0; font-size: 20px; font-weight: 500; margin: 0 0 15px 0; text-align: center;">Thoughtfully designed to support parents during life transitions</h3>', unsafe_allow_html=True)
st.markdown('<p style="color: #6b7280; font-size: 16px; font-style: italic; margin: 0; text-align: center;">Supporting families through every stage of parenthood</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Features row - using Streamlit columns with simple styling
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div style="text-align: center; padding: 15px; color: #5a7ba8; font-weight: 500;">🤖 Powered by AI</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div style="text-align: center; padding: 15px; color: #5a7ba8; font-weight: 500;">🔒 Privacy-First</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div style="text-align: center; padding: 15px; color: #5a7ba8; font-weight: 500;">🌟 Open Source</div>', unsafe_allow_html=True)

# Team credit - single line for reliability
st.markdown('<div style="text-align: center; margin: 30px 0;"><div style="background: linear-gradient(135deg, #2c5aa0 0%, #1e4080 100%); color: white; padding: 18px 35px; border-radius: 25px; display: inline-block; box-shadow: 0 3px 12px rgba(44, 90, 160, 0.25);"><span style="font-size: 16px; font-weight: 500;">Made with ❤️ by Team Kotori.</span></div></div>', unsafe_allow_html=True)

# Copyright - single line
st.markdown('<div style="text-align: center; margin: 20px 0; padding-top: 20px; border-top: 1px solid rgba(173, 216, 230, 0.3);"><p style="font-size: 13px; color: #9ca3af; margin: 0;">© 2024 Kotori.ai • All rights reserved • Built with Streamlit & LangGraph</p></div>', unsafe_allow_html=True)