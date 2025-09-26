import streamlit as st
import requests
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# FastAPI endpoint
API_URL = "http://localhost:8000"

st.title("📚 Document Q&A System")
st.write("Upload documents, search through them, and ask questions!")

# Sidebar for collection management
with st.sidebar:
    st.header("Collection Management")
    collection_name = st.text_input("Collection Name", value="my_collection")
    
    if st.button("Create Collection"):
        response = requests.post(
            f"{API_URL}/create_collection",
            json={"collection_name": collection_name}
        )
        if response.status_code == 200:
            st.success(f"Collection '{collection_name}' created successfully!")
        else:
            st.error(f"Error: {response.text}")
    
    if st.button("Delete Collection"):
        response = requests.delete(f"{API_URL}/collection/{collection_name}")
        if response.status_code == 200:
            st.success(f"Collection '{collection_name}' deleted successfully!")
        else:
            st.error(f"Error: {response.text}")

# Main area tabs
tab1, tab2, tab3 = st.tabs(["Upload Document", "Search", "Ask Questions"])

# Upload Document Tab
with tab1:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
    
    if uploaded_file is not None:
        # Save the file temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        if st.button("Process Document"):
            response = requests.post(
                f"{API_URL}/upload_file",
                json={
                    "file_name": uploaded_file.name,
                    "collection_name": collection_name
                }
            )
            if response.status_code == 200:
                st.success("Document processed and uploaded successfully!")
                st.json(response.json())
            else:
                st.error(f"Error: {response.text}")
            
            # Clean up the temporary file
            os.remove(uploaded_file.name)

# Search Tab
with tab2:
    st.header("Search Documents")
    search_query = st.text_input("Enter your search query")
    num_results = st.slider("Number of results", min_value=1, max_value=10, value=5)
    
    if st.button("Search") and search_query:
        response = requests.post(
            f"{API_URL}/search",
            json={
                "query": search_query,
                "collection_name": collection_name,
                "limit": num_results
            }
        )
        if response.status_code == 200:
            results = response.json()["results"]
            for idx, result in enumerate(results, 1):
                st.markdown(f"**Result {idx}** (Score: {result['score']:.4f})")
                st.write(result["text"])
                st.write(f"Source: {result['source']}")
                st.markdown("---")
        else:
            st.error(f"Error: {response.text}")

# Ask Questions Tab
with tab3:
    st.header("Ask Questions")
    question = st.text_input("Enter your question")
    model = st.selectbox(
        "Select Model",
        ["gemma2-9b-it", "qwen/qwen3-32b"]
    )
    
    if st.button("Ask") and question:
        with st.spinner("Generating answer..."):
            response = requests.post(
                f"{API_URL}/generate",
                json={
                    "question": question,
                    "collection_name": collection_name,
                    "model": model
                }
            )
            if response.status_code == 200:
                result = response.json()
                st.markdown("### Answer:")
                answer_text = result["response"]
                st.write(answer_text)
                
                st.markdown("### Source Documents:")
                for idx, doc in enumerate(result["source_documents"], 1):
                    with st.expander(f"Source {idx} (Score: {doc['score']:.4f})"):
                        st.write(doc["text"])
                        st.write(f"Source: {doc['source']}")
            else:
                st.error(f"Error: {response.text}")
