from embeddings import embeddita
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document 

# Update memory initialization
memory = ConversationBufferMemory(return_messages=True)  

PROMPT = """
You are an expert assistant. Answer the given question clearly, concisely, and directly using only the provided context. Be sure to respond accordingly if you are being greeted. Keep the answers crisp. Do NOT summarize the context. Instead, provide **direct, actionable insights**. Use **bullet points or numbered steps** where relevant.
Keep the response **short, to the point, and helpful**.

**Context:** 
{context}

---

**Question:** {question}

"""

def query_rag(query_text: str):
    embed = embeddita()
    db = Chroma(persist_directory='chroma', embedding_function=embed)

    # Retrieve relevant documents & past conversations
    results = db.similarity_search_with_score(query_text, k=5)
    retrieved_texts = [doc.page_content for doc, _ in results]

    # Retrieve past conversation history stored in Chroma
    past_results = db.similarity_search_with_score(f"CONVERSATION_HISTORY: {query_text}", k=5)
    past_texts = [doc.page_content for doc, _ in past_results]

    # Combine context from documents & past conversations
    context_text = "\n\n---\n\n".join(retrieved_texts + past_texts)

    # Construct the prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Generate response
    model = OllamaLLM(model="gemma2:2b", temperature=0.3, top_p=0.9)
    response_text = model.invoke(prompt)

    # Store conversation history in Chroma
    conversation_doc = Document(
        page_content=f"User: {query_text}\nAssistant: {response_text}",
        metadata={"source": "conversation_history", "id": f"query_{hash(query_text)}"}
    )
    db.add_documents([conversation_doc], ids=[conversation_doc.metadata["id"]])


    # Extract sources
    sources = [doc.metadata.get("id", None) for doc, _ in results]
    formatted_response = f"""
    
**Response:**
{response_text.strip()}

**Sources:**
{", ".join(str(src) for src in sources if src)}

"""
    print(formatted_response)
    return response_text

# Example query
response = query_rag("How to cope with Empty Nest Syndrome?")
