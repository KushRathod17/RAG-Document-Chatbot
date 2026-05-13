from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

#File loader
def load_pdf(file_path):
    loader=PyPDFLoader(file_path)
    pages=loader.load()
    print(f"Loaded{len(pages)}pages")
    return pages
#Splitting data into chuks and such that is doesnt miss any data
def split_into_chunks(pages):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=['\n\n','\n','.','']
    )
    chunks=splitter.split_documents(pages)
    print(f"Split into{len(chunks)}chunks")
    return chunks
def create_vector_store(chunks):
    print("Loading Embedding Model...")
    embeddings=HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    print("Creating vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="db"
    )
    print("Vector store created and saved")
    return vector_store 
#Main_code
if __name__ == "__main__":
    pages = load_pdf("sample_pdfs/sample_pdf1.pdf")
    chunks = split_into_chunks(pages)
    vector_store=create_vector_store(chunks) 
    print(f"Total chunks created: {len(chunks)}")
    
