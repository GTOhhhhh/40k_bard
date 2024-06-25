# main.py

import os
import json
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from vector_db import PERSIST_DIRECTORY
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Load API key from secrets.json
with open("./secrets.json") as f:
    secrets = json.load(f)
    os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]

# Set up Langchain components
llm = ChatOpenAI(model_name="gpt-4")
embeddings = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

# Create a custom prompt template
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer: """

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

def generate_response(query):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    result = qa_chain({"query": query})
    return result["result"], result["source_documents"]

def main():
    print("Welcome to the Warhammer 40,000 Query System!")
    print("Type 'exit' to quit the program.")

    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'exit':
            break

        answer, sources = generate_response(query)
        print(f"\nAnswer: {answer}")
        print("\nSources:")
        for i, doc in enumerate(sources, 1):
            print(f"{i}. {doc.metadata['source']}")

    print("Thank you for using the Warhammer 40,000 Query System!")

if __name__ == "__main__":
    main()