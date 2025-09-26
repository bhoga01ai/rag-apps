import os
import streamlit as st
import pickle
import time
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

st.title("ZionCloudSolutionsBot: Humanservices Tool")
st.sidebar.title("ZionCloudSolutions Human Services URLs")

urls = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()
if process_url_clicked:
    embeddings=OpenAIEmbeddings()

        # "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-8-2023",
        # "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-9-2023",
    
    # urls = [
    #     "https://hints20.livermoretemple.org/hints/yande/global.html"
    # ]
    
    main_placeholder.text("Data Loading...Started...✅✅✅")
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    time.sleep(2)

    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','],chunk_size=10000)
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    time.sleep(2)
    # print(docs)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    db = FAISS.from_documents(docs, embeddings)
    time.sleep(2)
    db.save_local('faiss_index')

    main_placeholder.text("Finished Indexing urls...✅✅✅")
    time.sleep(2)

query = main_placeholder.text_input("Question: ")
enter=st.button('Enter')

if query and enter:
    embeddings=OpenAIEmbeddings()
    db=FAISS.load_local('faiss_index',embeddings,allow_dangerous_deserialization=True)
    # query='who are the executive commitee members'
    results=db.similarity_search(query,k=10)
    # print((results[0].page_content))
    context=''
    for result in results:
        context=context+result.page_content
    print(context)

    prompt=''''
    Answer the following question based on the context and format the output for user friendly 
    question:{question}
    context:{context}
    '''.format(question=query,context=context)

    llm =  ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.5)

    response=llm.invoke(prompt)
    print(response.content)
    main_placeholder = st.empty()
    st.write(response.content)