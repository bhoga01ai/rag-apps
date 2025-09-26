# start qdrant vector database locally 
# docker pull qdrant/qdrant if doesn't exists in local
# docker run -p 6333:6333 -p 6334:6334 \
#     -v $(pwd)/qdrant_storage:/qdrant/storage:z \
#     qdrant/qdrant

import streamlit as st
import asyncio

asyncio.set_event_loop(asyncio.new_event_loop())
import time
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import SeleniumURLLoader
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd
import asyncio

def write_to_csv(feedback):
  """Writes feedback to a CSV file.

  Args:
    feedback: The feedback to write.
  """
  print('in fucntion')
  df = pd.DataFrame({'Feedback': [feedback]})
  df.to_csv('./feedback.csv', mode='a', index=False, header=not os.path.exists('./feedback.csv'))

vdb_client=QdrantClient(url="https://3bfb46b1-a7a7-4827-864a-7b1c8e9afe4b.us-east4-0.gcp.cloud.qdrant.io:6333",api_key=os.getenv('QDRANT_API_KEY'))
st.sidebar.title("ZionCloudSolutions Human Services URLs")
print(os.getenv('QDRANT_API_KEY'))
print("connect success")

urls = []
for i in range(1):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
if process_url_clicked:
    main_placeholder = st.sidebar.empty()
    embeddings=SentenceTransformer("all-MiniLM-L6-v2")

        # "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-8-2023",
        # "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-9-2023",

    
    main_placeholder.text("Data Loading...Started...✅✅✅")
    #loader = UnstructuredURLLoader(urls=urls)
    loader=SeleniumURLLoader(urls=urls)
    data = loader.load()
    time.sleep(2)

    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','],chunk_size=10000)
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    time.sleep(2)
    
    doc_chunks=[]
    for chunk in docs:
        chunk={
            "raw_text":chunk.page_content,
            "meta_data":chunk.metadata,
            "created_date":"2024-12-22"
        }
        doc_chunks.append(chunk)
    # print(doc_chunks)
    
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    vdb_client.create_collection(
        collection_name='vdb_index',
        vectors_config=models.VectorParams(
            size=embeddings.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE
        ),
    )
    vdb_client.upload_points(
        collection_name="vdb_index",
        points=[
            models.PointStruct(
                id=idx,vector=embeddings.encode(doc["raw_text"]).tolist(),payload=doc
            )
            for idx,doc in enumerate(doc_chunks)
        ]
    )
    time.sleep(2)
    main_placeholder.text("Finished Indexing urls...✅✅✅")
    time.sleep(2)

st.title("ZionCloudSolutionsBot: Humanservices Tool")
question = st.text_input("Question: ")
enter=st.button('Enter')

if question and enter:
    vdb_client=QdrantClient(url="https://3bfb46b1-a7a7-4827-864a-7b1c8e9afe4b.us-east4-0.gcp.cloud.qdrant.io:6333",api_key=os.getenv('QDRANT_API_KEY'))
    vdb_client.get_collections()
    embeddings=SentenceTransformer("all-MiniLM-L6-v2")
    results=vdb_client.query_points(
        collection_name="vdb_index",
        query=embeddings.encode(question).tolist(),
        limit=3
    ).points
    context=''
    sources=[]
    for result in results:
        print(result)
        context=context+result.payload['raw_text']
        if result.payload['meta_data']['source'] not in sources:
            sources.append([result.payload['meta_data']['source'],result.score])
    print(sources)

    prompt=''''
    Answer the following question based on the context and format the output for user friendly 
    question:{question}
    context:{context}
    '''.format(question=question,context=context)

    llm =  ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.5)

    response=llm.invoke(prompt)
    # print(response.content)
    st.write(response.content)   
    displayed_urls=[]
    for src in sources:
        if src[0] not in displayed_urls:
            displayed_urls.append(src[0]) 
            output=st.container()
            if output:
                st.write(str(src[0]) + '  score:' + str(src[1]))
            
            # Create the icons within the container
            feedback=st.container()
            if feedback:
                st.write("User Feedback:")
                with st.expander("Provide Feedback"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if st.button("👍"):
                            with st.spinner("Processing your feedback..."):
                                asyncio.create_task(write_to_csv("Positive Feedback"))
                            st.success("Thank you for your positive feedback!")
                    with col2:
                        if st.button("👎"):
                            asyncio.create_task(write_to_csv("Negative Feedback"))
                            st.write("Thank you for your positive feedback!")
                    with col3:
                        st.button("Share", type="primary")
                    with col4:
                        st.button("Refresh")
