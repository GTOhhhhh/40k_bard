import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

# Define the path for the persistent database
PERSIST_DIRECTORY = "./chroma_db"

def process_and_store_documents(file_path):
    # Load documents
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Initialize embeddings
    embeddings = OpenAIEmbeddings()

    # Create and persist the vector store
    vectordb = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings, 
        persist_directory=PERSIST_DIRECTORY
    )

    # Persist the database
    vectordb.persist()
    
    print(f"Processed and stored {len(texts)} document chunks in {PERSIST_DIRECTORY}")

def main():
    # Assume the document is in the same directory as the script
    document_path = "state_of_the_union.txt"
    
    if not os.path.exists(document_path):
        print(f"Error: {document_path} not found.")
        return

    process_and_store_documents(document_path)

    # Verify that we can load the persisted database
    loaded_vectordb = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=OpenAIEmbeddings()
    )

    # Simple test query
    query = "Who is Atlas' Jack?"
    docs = loaded_vectordb.similarity_search(query)
    print(f"\nTest query: '{query}'")
    print(f"Found {len(docs)} relevant document chunks")
    print(f"Sample result: {docs[0].page_content[:200]}...")

if __name__ == "__main__":
    main()