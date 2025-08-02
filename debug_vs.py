import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
if not hf_token:
    print("❌ HUGGINGFACE_API_TOKEN is missing.")
    exit(1)

def debug_vectorstore():
    """Debug function to check vectorstore contents and paths"""
    
    print("🔍 KOTORI.AI VECTORSTORE DEBUG REPORT")
    print("=" * 50)
    
    # Check both path configurations
    relative_path = Path("D:/Kotori.ai/chroma")
    hardcoded_path = Path("D:/Kotori.ai/chroma")
    
    print(f"📁 Checking paths:")
    print(f"   Relative path: {relative_path}")
    print(f"   Hardcoded path: {hardcoded_path}")
    print(f"   Relative exists: {relative_path.exists()}")
    print(f"   Hardcoded exists: {hardcoded_path.exists()}")
    
    # Try both paths
    for path_name, chroma_path in [("Relative", relative_path), ("Hardcoded", hardcoded_path)]:
        print(f"\n🔍 Testing {path_name} path: {chroma_path}")
        
        if not chroma_path.exists():
            print(f"❌ Path doesn't exist!")
            continue
            
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-base-en-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            
            vectorstore = Chroma(
                persist_directory=str(chroma_path), 
                embedding_function=embedding_model
            )
            
            # Check if vectorstore has any documents
            all_docs = vectorstore.get(include=["documents", "metadatas"])
            print(f"📊 Total documents: {len(all_docs['ids'])}")
            
            if len(all_docs['ids']) > 0:
                print(f"📄 First document preview: {all_docs['documents'][0][:200]}...")
                print(f"🏷️ First document metadata: {all_docs['metadatas'][0]}")
                
                # Test similarity search
                test_queries = [
                    "empty nest syndrome",
                    "children leaving home", 
                    "parenting after kids move out"
                ]
                
                for query in test_queries:
                    results = vectorstore.similarity_search(query, k=3)
                    print(f"🔍 Search '{query}': {len(results)} results")
                    if results:
                        print(f"   Top result: {results[0].page_content[:100]}...")
            else:
                print("❌ Vectorstore is empty!")
                
        except Exception as e:
            print(f"❌ Error with {path_name} path: {e}")
    
    # Check data directory
    print(f"\n📁 Checking data directory:")
    data_dir = Path("D:/Kotori.ai/data")
    print(f"   Data path: {data_dir}")
    print(f"   Data exists: {data_dir.exists()}")
    
    if data_dir.exists():
        pdf_files = list(data_dir.glob("*.pdf"))
        print(f"   PDF files found: {len(pdf_files)}")
        for pdf in pdf_files:
            print(f"     - {pdf.name} ({pdf.stat().st_size} bytes)")
    else:
        print("❌ Data directory doesn't exist!")

if __name__ == "__main__":
    debug_vectorstore()