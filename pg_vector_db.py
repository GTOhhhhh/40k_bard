import os
import json
import glob
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_postgres import PGVector
from sqlalchemy import create_engine

# Load API key from secrets.json
with open("./secrets.json") as f:
    secrets = json.load(f)
    os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]

# Define the PostgreSQL connection string
CONNECTION_STRING = "postgresql+psycopg://langchain:langchain@localhost:5432/langchain"
COLLECTION_NAME = "wh40k_docs"

def read_json_content(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return Document(page_content=data.get("content", ""), metadata={"source": file_path})


def process_and_store_documents(file_paths):
    documents = []
    for file_path in file_paths:
        documents.append(read_json_content(file_path))

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Initialize embeddings
    embeddings = OpenAIEmbeddings()

    # Create SQLAlchemy engine
    engine = create_engine(CONNECTION_STRING)

    # Create and persist the vector store using PGVector
    vectordb = PGVector.from_documents(
        documents=texts,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection=engine,
        use_jsonb=True,
    )

    print(f"Processed and stored {len(texts)} document chunks in PostgreSQL")


def main():
    # Get all JSON files in the wh40k_pages directory
    json_files = glob.glob("wh40k_pages/*.json")

    if not json_files:
        print("Error: No JSON files found in wh40k_pages directory.")
        return

    # Uncomment the following line to process and store documents
    process_and_store_documents(json_files)

    # Create SQLAlchemy engine
    engine = create_engine(CONNECTION_STRING)

    # Initialize PGVector for querying
    embeddings = OpenAIEmbeddings()
    vectordb = PGVector(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection=engine,
        use_jsonb=True,
    )


    # Simple test query
    query = "Who is 'Atlas' Jack?"
    docs = vectordb.similarity_search(query)
    print(f"\nTest query: '{query}'")
    print(f"Found {len(docs)} relevant document chunks")
    print(f"Sample result: {docs[0].page_content[:200]}...")


if __name__ == "__main__":
    main()
