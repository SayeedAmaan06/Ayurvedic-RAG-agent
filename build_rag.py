"""
build_rag.py
------------
Run this ONCE to read all documents in the data/ folder and build a
local Chroma vector store (saved to disk in chroma_db/).

This is intentionally separate from the agent code so students can see
the RAG "indexing" step and the "querying" step as two different things.

Usage:
    python build_rag.py
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_DIR = "data"
DB_DIR = "chroma_db"


def main():
    print("Loading documents from:", DATA_DIR)
    loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")

    # Split into small chunks so retrieval is more precise
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    # Free, local, no-PyTorch embedding model (keeps memory usage low,
    # important for free-tier hosting on Render).
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # This creates (or overwrites) the local vector store on disk
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    print(f"Vector store saved to: {DB_DIR}")


if __name__ == "__main__":
    main()
