import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    CSVLoader, JSONLoader, TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
)
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

INDEX_PATH = "../faiss_index"

def load_file(file_path):
    """
    Load a file based on its extension and return the documents.
    Supported formats: CSV, JSON, PDF, TXT, DOCX.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        loader = CSVLoader(file_path=file_path, encoding="utf-8")
    elif ext == ".json":
        loader = JSONLoader(file_path=file_path, jq_schema=".", text_content=False)
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext in [".doc", ".docx"]:
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()

def preprocess_docs(documents, chunk_size=500, chunk_overlap=50):
    """
    Preprocess documents by splitting them into smaller chunks.
    This is useful for creating a vector store.
    """
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def create_or_update_vector_store(docs, index_path=INDEX_PATH):
    """
    Create or update the FAISS vector store with the provided documents.
    If the index already exists, it will add new documents to it.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )

    if os.path.exists(os.path.join(index_path, "index.faiss")):
        print("Loading existing FAISS index...")
        vectorstore = FAISS.load_local(index_path, embeddings)
        vectorstore.add_documents(docs)
        print("New documents added to existing index.")
    else:
        print("Creating new FAISS index...")
        vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local(index_path)
    print(f"Vector store saved to {index_path}.")
    return vectorstore

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    raw_docs = load_file(file_path)
    processed_docs = preprocess_docs(raw_docs)
    create_or_update_vector_store(processed_docs)

if __name__ == "__main__":
    main()
