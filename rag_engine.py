import hashlib
import os
import chromadb
from dotenv import load_dotenv

from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

load_dotenv()

# ======================================================================
#  CONFIGURATION (Ollama only — local & free)
# ======================================================================
# Models are configurable via env vars, with sensible defaults.
OLLAMA_LLM = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_EMBED = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

print("=" * 60)
print("⚙️  INITIALISING RAG ENGINE (Ollama / local)")
print(f"🦙 LLM model:        {OLLAMA_LLM}")
print(f"🔢 Embedding model:  {OLLAMA_EMBED}")

# The model that generates answers (runs locally via Ollama)
Settings.llm = Ollama(model=OLLAMA_LLM, request_timeout=120.0)

# The model that creates embeddings (runs locally via Ollama)
Settings.embed_model = OllamaEmbedding(model_name=OLLAMA_EMBED)

# How the text is split into chunks
Settings.node_parser = SentenceSplitter(chunk_size=600, chunk_overlap=80)
print("⚙️  Chunking: size=600, overlap=80")
print("=" * 60)

# ======================================================================
#  HELPERS
# ======================================================================

def _get_collection():
    """Open (or create) the Chroma collection. The name includes the embedding
    model, because embeddings from different models have different dimensions
    and are NOT compatible with each other.

    :return: The Chroma collection object.
    """
    db = chromadb.PersistentClient(path="./chroma_db")
    # Embedding-model-specific name avoids dimension mismatch if you ever switch embedding models.
    safe_name = OLLAMA_EMBED.replace("-", "_").replace(":", "_")
    collection_name = f"collection_{safe_name}"
    collection = db.get_or_create_collection(collection_name)
    print(
        f"📦 [Chroma] Using collection '{collection_name}' "
        f"({collection.count()} chunks currently stored)"
    )
    return collection


def compute_file_hash(file_path: str) -> str:
    """Compute a SHA-256 hash of a file's CONTENT (not its name).

    :param file_path: Path to the file on disk.
    :return: The hex digest string.
    """
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha.update(block)
    digest = sha.hexdigest()
    print(f"🔑 [Hash] {os.path.basename(file_path)} -> {digest[:12]}...")
    return digest


def get_indexed_file_info(chroma_collection) -> dict:
    """Build a map of {file_name: set_of_hashes} from the metadata stored on
    every chunk.

    :param chroma_collection: The Chroma collection object.
    :return: dict mapping file names to the set of stored hashes.
    """
    info = {}
    if chroma_collection.count() == 0:
        print("ℹ️  [Index] Collection is empty (no files indexed yet)")
        return info

    data = chroma_collection.get(include=["metadatas"])
    for md in data["metadatas"]:
        if not md:
            continue
        name = md.get("file_name")
        h = md.get("file_hash")
        if name:
            info.setdefault(name, set()).add(h)

    print(f"ℹ️  [Index] Already indexed files: {list(info.keys())}")
    return info


def _make_unique_name(folder: str, file_name: str) -> str:
    """Generate a non-clashing name like 'report_1.pdf', 'report_2.pdf'...

    :param folder: The documents folder.
    :param file_name: The original (clashing) file name.
    :return: A new unique file name.
    """
    existing_names = set(get_indexed_file_info(_get_collection()).keys())
    base, ext = os.path.splitext(file_name)
    candidate = file_name
    i = 1
    while candidate in existing_names or os.path.exists(
        os.path.join(folder, candidate)
    ):
        candidate = f"{base}_{i}{ext}"
        i += 1
    print(f"🔤 [Rename] '{file_name}' -> '{candidate}'")
    return candidate


def _embed_one_file(path: str, file_name: str, file_hash: str, storage_context):
    """Read a single file, attach metadata, then embed and store it.

    :param path: Path to the file on disk.
    :param file_name: The name to store in the metadata.
    :param file_hash: The content hash to store in the metadata.
    :param storage_context: The StorageContext pointing at Chroma.
    :return: None
    """
    print(f"\n📄 [Parse] Reading '{file_name}'...")
    docs = SimpleDirectoryReader(input_files=[path]).load_data()
    print(f"📄 [Parse] '{file_name}' -> {len(docs)} document section(s)")

    # Attach metadata to every chunk
    for d in docs:
        d.metadata["file_name"] = file_name
        d.metadata["file_hash"] = file_hash

    print(
        f"🔢 [Embed] Creating embeddings & storing in Chroma "
        f"(runs locally via Ollama)..."
    )
    VectorStoreIndex.from_documents(
        documents=docs, storage_context=storage_context
    )
    print(f"✅ [Embed] '{file_name}' stored successfully")


# ======================================================================
#  CLASSIFY  ->  PROCESS
# ======================================================================


def classify_files(documents_folder: str = "data"):
    """Classify each file WITHOUT embedding:

    - 'to_skip'   : identical content already indexed (same hash)
    - 'conflicts' : same NAME exists but content is DIFFERENT (new hash)
    - 'to_add'    : brand-new file

    :param documents_folder: Folder with the uploaded files.
    :return: (classification dict, file_hashes dict {name: hash}).
    """
    print(f"\n{'=' * 60}")
    print(f"🔍 [Classify] Scanning folder '{documents_folder}'...")

    coll = _get_collection()
    file_info = get_indexed_file_info(coll)
    all_hashes = set().union(*file_info.values()) if file_info else set()

    classification = {"to_add": [], "to_skip": [], "conflicts": []}
    file_hashes = {}

    for file_name in os.listdir(documents_folder):
        path = os.path.join(documents_folder, file_name)
        if not os.path.isfile(path):
            continue

        h = compute_file_hash(path)
        file_hashes[file_name] = h

        if h in all_hashes:
            classification["to_skip"].append(file_name)
            print(f"   ⏭️  '{file_name}' -> SKIP (content already indexed)")
        elif file_name in file_info:
            classification["conflicts"].append(file_name)
            print(f"   ⚠️  '{file_name}' -> CONFLICT (same name, new content)")
        else:
            classification["to_add"].append(file_name)
            print(f"   🆕 '{file_name}' -> ADD (brand new)")

    print(
        f"🔍 [Classify] Summary: "
        f"{len(classification['to_add'])} to add, "
        f"{len(classification['conflicts'])} conflicts, "
        f"{len(classification['to_skip'])} to skip"
    )
    print("=" * 60)
    return classification, file_hashes


def process_files(
    documents_folder, classification, file_hashes, decisions: dict
):
    """Apply the actual work based on classification and user decisions.
    This is where embeddings happen.

    :param documents_folder: The documents folder.
    :param classification: dict from classify_files().
    :param file_hashes: dict mapping file name -> hash.
    :param decisions: dict {file_name: "replace" | "keep_both"} for conflicts.
    :return: A report dict describing what happened.
    """
    print(f"\n{'=' * 60}")
    print("⚙️  [Process] Starting embedding work...")

    coll = _get_collection()
    vector_store = ChromaVectorStore(chroma_collection=coll)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    report = {
        "added": [],
        "skipped": list(classification["to_skip"]),
        "replaced": [],
        "kept_both": [],
    }

    # Brand-new files
    for fn in classification["to_add"]:
        path = os.path.join(documents_folder, fn)
        _embed_one_file(path, fn, file_hashes[fn], storage_context)
        report["added"].append(fn)

    # Conflicts -> follow the user's choice
    for fn in classification["conflicts"]:
        choice = decisions.get(fn, "replace")
        path = os.path.join(documents_folder, fn)
        print(f"\n⚖️  [Conflict] '{fn}' -> decision: {choice.upper()}")

        if choice == "replace":
            print(f"🗑️  [Replace] Deleting old chunks of '{fn}'...")
            coll.delete(where={"file_name": fn})
            _embed_one_file(path, fn, file_hashes[fn], storage_context)
            report["replaced"].append(fn)
        else:  # keep_both
            new_name = _make_unique_name(documents_folder, fn)
            new_path = os.path.join(documents_folder, new_name)
            os.rename(path, new_path)
            _embed_one_file(
                new_path, new_name, file_hashes[fn], storage_context
            )
            report["kept_both"].append(f"{fn} -> {new_name}")

    print(f"\n⚙️  [Process] Done. Report: {report}")
    print("=" * 60)
    return report


# ======================================================================
#  LOAD  &  CHAT
# ======================================================================

def load_index():
    """Load the existing index from Chroma WITHOUT re-embedding anything.

    :return: A VectorStoreIndex if the collection has data, else None.
    """
    coll = _get_collection()
    if coll.count() == 0:
        print("⚠️  [Load] Nothing to load — collection is empty")
        return None
    vector_store = ChromaVectorStore(chroma_collection=coll)
    index = VectorStoreIndex.from_vector_store(vector_store)
    print(f"✅ [Load] Loaded index with {coll.count()} chunks")
    return index


def get_chat_engine(index):
    """Build a basic chat engine (vector search only) with conversational
    memory.

    :param index: The VectorStoreIndex created/loaded earlier.
    :return: A chat engine exposing a .chat(message) method.
    """
    print("💬 [Engine] Building BASIC chat engine (vector only, top_k=3)")
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        similarity_top_k=3,
        verbose=False,
    )
    return chat_engine


def _load_nodes_from_chroma():
    """Pull every stored chunk out of Chroma and rebuild as TextNode objects.
    BM25 needs the text in memory. We reuse Chroma IDs so fusion (RRF) can
    deduplicate correctly.

    :return: A list of TextNode objects, or [] if the collection is empty.
    """
    coll = _get_collection()
    if coll.count() == 0:
        return []

    data = coll.get(include=["documents", "metadatas"])
    nodes = []
    for node_id, text, md in zip(
        data["ids"], data["documents"], data["metadatas"]
    ):
        nodes.append(TextNode(text=text, metadata=md or {}, id_=node_id))

    print(
        f"🔁 [BM25] Reconstructed {len(nodes)} nodes from Chroma "
        f"(needed for keyword search)"
    )
    return nodes


def get_hybrid_chat_engine(
    index, top_k_retrieve: int = 10, top_n_rerank: int = 3
):
    """Build an advanced chat engine: hybrid search (vector + BM25) + reranker.

    :param index: The VectorStoreIndex (loaded from Chroma).
    :param top_k_retrieve: How many candidates each retriever pulls.
    :param top_n_rerank: How many chunks survive after reranking.
    :return: A chat engine, or None if there are no documents indexed yet.
    """
    print(
        f"\n🚀 [Engine] Building HYBRID chat engine "
        f"(retrieve top-{top_k_retrieve}, rerank to top-{top_n_rerank})"
    )

    nodes = _load_nodes_from_chroma()
    if not nodes:
        print("⚠️  [Engine] No nodes available — cannot build hybrid engine")
        return None

    # 1) Dense retriever (semantic / vector search)
    vector_retriever = index.as_retriever(similarity_top_k=top_k_retrieve)
    print("   ✓ Vector retriever ready (semantic search)")

    # 2) Sparse retriever (BM25 / keyword search)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k_retrieve,
    )
    print("   ✓ BM25 retriever ready (keyword search)")

    # 3) Hybrid: fuse both with Reciprocal Rank Fusion
    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=top_k_retrieve,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )
    print("   ✓ Fusion retriever ready (Reciprocal Rank Fusion)")

    # 4) Reranker: local cross-encoder
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=top_n_rerank,
    )
    print("   ✓ Cross-encoder reranker ready")

    # 5) Wrap in a chat engine with memory
    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=hybrid_retriever,
        node_postprocessors=[reranker],
        verbose=False,
    )
    print("✅ [Engine] Hybrid chat engine ready!")
    return chat_engine