from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector

embeddings = OpenAIEmbeddings()

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = TextLoader("sotu_address_obama.txt", encoding='utf-8')
documents = loader.load()

print(documents)  # prints the document objects
print(len(documents))  # 1 - we've only read one file/document into the loader

# split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,separators=['\n\n', '\n', '.', ','],
)
docs = text_splitter.split_documents(documents)

print(docs)  # prints the document objects
print(len(docs))  # 1 - we've only read one file/document into the loader

CONNECTION_STRING = f"postgresql+psycopg2://postgres:mysecretpassword@localhost:5432/vector_db"
COLLECTION_NAME = "sotu_speech_99"

db = PGVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    distance_strategy="cosine",
    pre_delete_collection=True
)
print(db)

query = "What did the president say about higher education"
similar_docs = db.similarity_search_with_score(query, k=3)

print(f"\nSearch Query: '{query}'\n")
print("Similar Documents Found:")
print("-" * 80)

for i, (doc, score) in enumerate(similar_docs, 1):
    print(f"\nDocument {i} (Similarity Score: {score:.4f})")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"Content: {doc.page_content}")
    print("-" * 80)