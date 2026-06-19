# build_baseline_2books.py
"""Build a NORMAL index for the same 2 books (for fair comparison)."""

import chromadb
import os
from dotenv import load_dotenv
from rag_engine import EMBED_NAME, SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

BOOKS = [
    "A Brief History of Time by Stephen Hawking.pdf",
    "Flux J. Machine Learning Mathematics in Python 2024.pdf",
    "Paxton F. Foundations of Naturopathic Nutrition. A Comprehensive Guide..2ed 2025.pdf",
    "Six Easy Pieces_ Essentials of Physics Explained by Its Most Brilliant Teacher by Richard P. Feynman.pdf",
    "Theodoridis S. Machine Learning From the Classics to Deep Networks,..3ed 2025.pdf",
    "Advanced Nutrition, 3rd Edition.pdf",
]

db = chromadb.PersistentClient(path="./chroma_db")
safe = EMBED_NAME.replace("-", "_").replace(":", "_")
coll = db.get_or_create_collection(f"collection_{safe}_2books")
vs = ChromaVectorStore(chroma_collection=coll)
sc = StorageContext.from_defaults(vector_store=vs)

for fn in BOOKS:
    docs = SimpleDirectoryReader(input_files=[os.path.join("data", fn)]).load_data()
    for d in docs:
        d.metadata["file_name"] = fn
    VectorStoreIndex.from_documents(docs, storage_context=sc)
    print(f"✅ {fn}")

print(f"Total: {coll.count()} chunks")