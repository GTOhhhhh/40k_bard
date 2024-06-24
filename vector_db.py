import os
import json
import glob
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load API key from secrets.json
with open("./secrets.json") as f:
    secrets = json.load(f)
    os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]

# Define the path for the persistent database
PERSIST_DIRECTORY = "./chroma_db"


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

    # Create and persist the vector store
    vectordb = Chroma.from_documents(
        documents=texts, embedding=embeddings, persist_directory=PERSIST_DIRECTORY
    )

    # Persist the database
    vectordb.persist()

    print(f"Processed and stored {len(texts)} document chunks in {PERSIST_DIRECTORY}")


def main():
    # Get all JSON files in the wh40k_pages directory
    json_files = glob.glob("wh40k_pages/*.json")

    if not json_files:
        print("Error: No JSON files found in wh40k_pages directory.")
        return

    process_and_store_documents(json_files)

    # Verify that we can load the persisted database
    loaded_vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=OpenAIEmbeddings())

    # Simple test query
    query = "Who is 'Atlas' Jack?"
    docs = loaded_vectordb.similarity_search(query)
    print(f"\nTest query: '{query}'")
    print(f"Found {len(docs)} relevant document chunks")
    print(f"Sample result: {docs[0].page_content[:200]}...")


if __name__ == "__main__":
    main()
