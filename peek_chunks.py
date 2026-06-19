# peek_chunks.py
"""Peek at the actual content of stored chunks (for curiosity/debugging)."""

import chromadb
from rag_engine import EMBED_NAME

def peek(suffix, n=5, search=None):
    """
    Print the first n chunks from a collection.

    :param suffix: Collection suffix (e.g. 'sentence', 'contextual').
    :param n: How many chunks to show.
    :param search: Optional keyword to filter chunks containing it.
    """
    db = chromadb.PersistentClient(path="./chroma_db")
    safe = EMBED_NAME.replace("-", "_").replace(":", "_")
    coll_name = f"collection_{safe}_{suffix}"
    coll = db.get_or_create_collection(coll_name)

    print(f"\n{'=' * 70}")
    print(f"📦 Collection: {coll_name}  ({coll.count()} chunks)")
    print('=' * 70)

    data = coll.get(include=["documents", "metadatas"])
    docs = data["documents"]
    metas = data["metadatas"]

    shown = 0
    for i, (text, md) in enumerate(zip(docs, metas)):
        # Optional keyword filter
        if search and search.lower() not in text.lower():
            continue

        fname = (md or {}).get("file_name", "?")
        print(f"\n--- Chunk #{i} | 📄 {fname} ---")
        print(text[:500])
        print(f"... [{len(text)} chars total]")

        shown += 1
        if shown >= n:
            break

if __name__ == "__main__":
    # Show first 3 chunks from the contextual collection
    peek("contextual", n=3)

    # Compare: same area in the normal collection
    # peek("sentence", n=3)

    # Or search for a specific term:
    # peek("contextual", n=3, search="gradient descent")