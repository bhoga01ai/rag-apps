# qdrant_api.py - This is a FastAPI application that provides a REST API for interacting with Qdrant vector database.
# and was created by Venkat using Vibe Coding technology.

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import models, QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
import json
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Pydantic models for request payloads
class CollectionRequest(BaseModel):
    collection_name: str

class GenerateRequest(BaseModel):
    question: str
    collection_name: str
    limit: Optional[int] = 5
    model: Optional[str] = "llama3-70b-8192"

class FileUploadRequest(BaseModel):
    file_name: str
    collection_name: str

class SearchRequest(BaseModel):
    query: str
    collection_name: str
    limit: Optional[int] = 5

# Initialize Qdrant client
qdrant_url = "https://abae8011-6e7d-4331-9275-141914068878.us-east4-0.gcp.cloud.qdrant.io:6333"
qdrant_api_key = os.getenv("QDRANT_API_KEY")
vdb_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

@app.post("/create_collection")
async def create_collection(collection_request: CollectionRequest):
    try:
        # Create collection
        vdb_client.create_collection(
            collection_name=collection_request.collection_name,
            vectors_config=models.VectorParams(
                size=1536,  # OpenAI embeddings dimension
                distance=models.Distance.COSINE
            )
        )
        return {"message": f"Collection {collection_request.collection_name} created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(request: SearchRequest):
    try:
        # Get embeddings for the query
        query_vector = embeddings.embed_query(request.query)
        
        # Search in Qdrant
        search_result = vdb_client.search(
            collection_name=request.collection_name,
            query_vector=query_vector,
            limit=request.limit
        )
        
        # Format results
        results = []
        for scored_point in search_result:
            results.append({
                'id': scored_point.id,
                'score': scored_point.score,
                'text': scored_point.payload.get('text', ''),
                'source': scored_point.payload.get('source', ''),
                'directory': scored_point.payload.get('directory', '')
            })
            
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_file")
async def upload_file(request: FileUploadRequest):
    try:
        print(request)
        # Load and process the document
        loader = TextLoader(request.file_name, encoding='utf-8')
        # Split text into chunks    
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100)
        docs = text_splitter.split_documents(loader.load())
        print("in")
        # Format documents
        docs_formatted = []
        for idx, doc in enumerate(docs):
            docs_formatted.append({
                'id': idx + 1,
                'text': doc.page_content,
                'directory': os.getcwd(),
                'source': request.file_name
            })

        # Create points for Qdrant
        points = []
        for doc in docs_formatted:
            vector = embeddings.embed_query(doc['text'])
            point = models.PointStruct(
                id=doc['id'],
                vector=vector,
                payload=doc
            )
            points.append(point)

        # Upload points to Qdrant
        vdb_client.upload_points(
            collection_name=request.collection_name,
            points=points
        )

        return {
            "message": f"File processed and uploaded to collection {request.collection_name}",
            "documents_processed": len(points)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_response(request: GenerateRequest):
    try:
        # First, get relevant context using search
        search_request = SearchRequest(
            query=request.question,
            collection_name=request.collection_name,
            limit=request.limit
        )
        
        # Search for relevant context
        search_results = await search(search_request)
        
        # Combine all relevant text into context
        context = "\n".join([result["text"] for result in search_results["results"]])
        
        # Create prompt with the retrieved context
        prompt = f'''
        Use the following context to answer the question. If the answer cannot be found in the context, say "I don't have enough information to answer this question."
        
        Context:
        {context}
        
        Question: {request.question}
        
        Please provide a clear and concise answer based on the given context.
        '''
        
        # Generate response using the specified model
        llm = ChatGroq(model=request.model, temperature=0.5)
        response = llm.invoke(prompt)
        
        return {
            "response": response.content,
            "source_documents": search_results["results"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/collection/{collection_name}")
async def delete_collection(collection_name: str):
    try:
        vdb_client.delete_collection(collection_name=collection_name)
        return {"message": f"Collection {collection_name} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)