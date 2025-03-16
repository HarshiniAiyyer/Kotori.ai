import streamlit as st
from drafts.querydata import query_rag  

# Page Configuration
st.set_page_config(page_title="Kotori.ai", layout="wide")

# Apply Custom CSS
st.markdown(
    """
    <style>
    /* Main Content Background */
    .stApp {
        background-color: #bdd1f8 !important; /* Soft Pastel Blue */
    }

    @font-face {
        font-family: 'Intuitive';
        src: url('assets/fonts/intuitive.ttf') format('truetype');
    }

    /* Force Apply Intuitive Font to Headings */
    h1, h2, h3, h4, h5, h6, .stApp h1 {
        font-family: 'Intuitive', sans-serif !important;
        color: #1D3557 !important; /* Deep Blue */
    }

    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #98C8E8 !important; /* Gentle Sky Blue */
    }

    /* Text inside Sidebar */
    [data-testid="stSidebar"] * {
        color: #003366 !important;  /* Dark Blue */
    }

    /* Image Border */
    img {
        border: 3px solid #1D3557 !important;
        border-radius: 8px; /* Softens edges */
    }

    /* Clear Chat History Button */
    [data-testid="stSidebar"] button {
        background-color: white !important;
        color: #003366 !important;
        border: 2px solid #003366 !important;
    }

    /* Change Text Input Label (Query Title) */
        label[for="💬 Ask me anything:"] {
            font-size: 18px !important;
            color: #1D3557 !important; /* Deep Blue */
            font-family: Georgia, serif !important;
        }

    /* Button Hover Effect */
    [data-testid="stSidebar"] button:hover {
        background-color: #003366 !important;
        color: white !important;
    }

    /* Search Bar */
    input[type="text"] {
        background-color: white !important;
        color: #003366 !important;
    }

    /* Search Bar Placeholder */
    input::placeholder {
        color: #003366 !important;
        opacity: 1 !important; /* Fully visible */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Content
with st.sidebar:
    st.image("assets/images/image.png", use_container_width=True)  # Replace with actual image path
    st.subheader("About")
    st.write("This chatbot retrieves and answers questions using RAG.")

    # Chat History Section
    st.subheader("History")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for chat in reversed(st.session_state["chat_history"]):  # Show latest first
        with st.expander(f" {chat['query']}"):
            st.markdown(f"**Q:** {chat['query']}")
            st.markdown(f"**A:** {chat['response']}")

    # Clear Chat History Button
    if st.button("Clear Chat History?"):
        st.session_state["chat_history"] = []

# Main Page Content
st.title("Missing your little ones? Kotori.ai will help you diagnose it.")

# User Query Input
query = st.text_input("💬 Ask me anything:", placeholder="Type your query here...")

# Handle Query Response
if query:
    with st.spinner("Searching and generating response..."):
        # Simulated Response (Replace with actual function call)
        response = f"Here's my response to '{query}'..."
        
        # Store in session
        st.session_state["chat_history"].append({"query": query, "response": response})
    
    # Display Response
    st.subheader("Answer:")
    st.write(response)

    # Clear input field after submission
    st.session_state["last_query"] = query  # Store the last query
    st.rerun()  # Rerun the app to clear input
