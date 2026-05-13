from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv("rag.env")

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )
    return vector_store

def create_rag_chain(vector_store):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate.from_template("""
Use the following context to answer the question.
If you don't know the answer, say you don't know.

Context: {context}
Question: {question}
Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

def ask_question(chain, retriever, question):
    answer = chain.invoke(question)
    docs = retriever.invoke(question)
    
    print(f"\nAnswer: {answer}")
    print("\nSources:")
    for doc in docs:
        print(f"  - Page {doc.metadata['page']}: {doc.page_content[:100]}...")

if __name__ == "__main__":
    print("Loading vector store...")
    vector_store = load_vector_store()

    print("Creating RAG chain...")
    chain, retriever = create_rag_chain(vector_store)

    print("\nRAG Chatbot Ready! Type 'exit' to quit\n")
    while True:
        question = input("Your question: ")
        if question.lower() == "exit":
            break
        ask_question(chain, retriever, question)