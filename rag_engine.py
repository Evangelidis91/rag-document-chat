import hashlib
import os
import re
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
from llama_index.core.vector_stores import (
    MetadataFilter, MetadataFilters, FilterCondition
)

from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from llama_index.core.node_parser import SemanticSplitterNodeParser

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_LLM = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_EMBED = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

print("=" * 60)
print(f"⚙️  INITIALISING RAG ENGINE — provider: {PROVIDER.upper()}")

if PROVIDER == "openai":
    LLM_NAME = "gpt-4o-mini"
    EMBED_NAME = "text-embedding-3-small"
    Settings.llm = OpenAI(model=LLM_NAME, temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model=EMBED_NAME)
    print(f"☁️  LLM: {LLM_NAME}  |  Embeddings: {EMBED_NAME}")
else:
    LLM_NAME = OLLAMA_LLM
    EMBED_NAME = OLLAMA_EMBED
    Settings.llm = Ollama(model=OLLAMA_LLM, request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(model_name=OLLAMA_EMBED)
    print(f"🦙 LLM: {LLM_NAME}  |  Embeddings: {EMBED_NAME}")

CHUNKING = os.getenv("CHUNKING", "sentence").lower()

if CHUNKING == "semantic":
    Settings.node_parser = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=Settings.embed_model,
    )
    print(f"✂️  Chunking: SEMANTIC (breakpoint=95%)")
else:
    Settings.node_parser = SentenceSplitter(chunk_size=600, chunk_overlap=80)
    print(f"✂️  Chunking: SENTENCE (size=600, overlap=80)")

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions ONLY using the "
    "provided context from the user's documents.\n"
    "Rules:\n"
    "1. If the answer IS in the context, answer clearly and concisely.\n"
    "2. If the answer is NOT in the context, reply EXACTLY with: "
    "\"I couldn't find the answer to that in the provided documents.\"\n"
    "3. Never use outside knowledge or invent information.\n"
    "4. If a question is unrelated to the documents, politely say it is "
    "outside the scope of the loaded documents.\n"
    "5. Write math and formulas in plain text (e.g. 'f(x)', 'R^n'), "
    "NOT in LaTeX notation. Avoid backslash commands like \\mathbb."
)


def _get_collection():
    db = chromadb.PersistentClient(path="./chroma_db")
    safe_embed = EMBED_NAME.replace("-", "_").replace(":", "_")
    collection_name = f"collection_{safe_embed}_{CHUNKING}"
    collection = db.get_or_create_collection(collection_name)
    print(f"📦 [Chroma] Using collection '{collection_name}' "
          f"({collection.count()} chunks currently stored)")
    return collection


def compute_file_hash(file_path: str) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha.update(block)
    digest = sha.hexdigest()
    print(f"🔑 [Hash] {os.path.basename(file_path)} -> {digest[:12]}...")
    return digest


def get_indexed_file_info(chroma_collection) -> dict:
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
    print(f"\n📄 [Parse] Reading '{file_name}'...")
    docs = SimpleDirectoryReader(input_files=[path]).load_data()
    print(f"📄 [Parse] '{file_name}' -> {len(docs)} document section(s)")

    for d in docs:
        cleaned = d.get_content().replace("-\n", "").replace("- ", "")
        d.set_content(cleaned)
        d.metadata["file_name"] = file_name
        d.metadata["file_hash"] = file_hash

    print(f"🔢 [Embed] Creating embeddings & storing in Chroma...")
    VectorStoreIndex.from_documents(documents=docs, storage_context=storage_context)
    print(f"✅ [Embed] '{file_name}' stored successfully")


def _build_filter(file_names):
    if not file_names:
        return None
    filters = MetadataFilters(
        filters=[MetadataFilter(key="file_name", value=fn) for fn in file_names],
        condition=FilterCondition.OR,
    )
    return filters


def get_index_stats():
    coll = _get_collection()
    total_chunks = coll.count()
    per_file = {}
    if total_chunks > 0:
        data = coll.get(include=["metadatas"])
        for md in data["metadatas"]:
            if md and md.get("file_name"):
                name = md["file_name"]
                per_file[name] = per_file.get(name, 0) + 1

    stats = {
        "total_documents": len(per_file),
        "total_chunks": total_chunks,
        "per_file": per_file,
        "llm_model": LLM_NAME,
        "embed_model": EMBED_NAME,
    }
    return stats


def grade_chunks(question: str, nodes) -> str:
    if not nodes:
        return "irrelevant"
    context = "\n---\n".join(n.text for n in nodes[:3])
    prompt = (
        "You are a relevance grader. Decide if the CONTEXT contains "
        "information RELATED to the QUESTION's topic.\n"
        "Be LENIENT: if the context discusses the same subject, it counts "
        "as relevant even if it's not a perfect definition.\n"
        "Reply with ONE word: 'relevant' or 'irrelevant'.\n\n"
        f"QUESTION: {question}\n\n"
        f"CONTEXT:\n{context[:2000]}\n\n"
        "Grade:"
    )
    response = str(Settings.llm.complete(prompt)).strip().lower()
    print(f"⚖️  [CRAG] '{question[:30]}' -> {response}")
    if "irrelevant" in response:
        return "irrelevant"
    return "relevant"


def get_crag_query_engine(index, file_names=None, use_hybrid=True, top_k_retrieve=10, top_n_rerank=3):
    print(f"\n🔄 [CRAG] Building CRAG query engine (hybrid={use_hybrid})")
    if use_hybrid:
        nodes = _load_nodes_from_chroma(file_names=file_names)
        if not nodes:
            return None
        bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k_retrieve)
        filters = _build_filter(file_names)
        vector = index.as_retriever(similarity_top_k=top_k_retrieve, filters=filters)
        retriever = QueryFusionRetriever(
            [vector, bm25], similarity_top_k=top_k_retrieve,
            num_queries=1, mode="reciprocal_rerank", use_async=False,
        )
        reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=top_n_rerank,
        )
        postprocessors = [reranker]
    else:
        filters = _build_filter(file_names)
        retriever = index.as_retriever(similarity_top_k=top_n_rerank, filters=filters)
        postprocessors = []

    return CRAGQueryEngine(retriever, postprocessors)


def add_contextual_documents(documents_folder: str, only_files: list, collection_suffix: str = "contextual"):
    import time
    db = chromadb.PersistentClient(path="./chroma_db")
    safe_embed = EMBED_NAME.replace("-", "_").replace(":", "_")
    coll_name = f"collection_{safe_embed}_{collection_suffix}"
    coll = db.get_or_create_collection(coll_name)
    print(f"📦 [Contextual] Collection '{coll_name}' ({coll.count()} chunks already stored)")

    vector_store = ChromaVectorStore(chroma_collection=coll)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    splitter = SentenceSplitter(chunk_size=600, chunk_overlap=80)

    total = 0
    for file_name in only_files:
        path = os.path.join(documents_folder, file_name)
        if not os.path.isfile(path):
            print(f"⚠️  [Contextual] SKIPPING missing file: {file_name}")
            continue

        print(f"\n📄 [Contextual] Processing '{file_name}'...")
        docs = SimpleDirectoryReader(input_files=[path]).load_data()
        nodes = splitter.get_nodes_from_documents(docs)
        print(f"📄 [Contextual] {len(nodes)} chunks to contextualise...")

        contextualised = []
        for i, node in enumerate(nodes):
            chunk_text = node.get_content()
            context_prompt = (
                "Give a SHORT context (one sentence) situating this chunk "
                "within its document, to improve search retrieval. "
                "State the topic or section it belongs to. Be concise.\n\n"
                f"Document: {file_name}\n"
                f"Chunk:\n{chunk_text[:800]}\n\n"
                "Short context:"
            )

            header = ""
            for attempt in range(3):
                try:
                    header = str(Settings.llm.complete(context_prompt)).strip()
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"   ⚠️ [Contextual] Retry {attempt + 1}/3: {e}")
                    time.sleep(wait)
            else:
                print(f"   ❌ [Contextual] Gave up on chunk {i} — using empty header")

            new_text = f"[Context: {header}]\n{chunk_text}"
            node.set_content(new_text)
            node.metadata["file_name"] = file_name
            node.metadata["contextualised"] = True
            contextualised.append(node)

            time.sleep(0.05)
            if (i + 1) % 50 == 0:
                print(f"   ...{i + 1}/{len(nodes)} chunks done")

        print(f"🔢 [Contextual] Embedding {len(contextualised)} chunks...")
        VectorStoreIndex(contextualised, storage_context=storage_context)
        total += len(contextualised)
    return total


def add_contextual_from_existing(source_suffix: str = "sentence", target_suffix: str = "contextual"):
    import hashlib
    import json
    import time
    from llama_index.core.schema import TextNode

    safe_embed = EMBED_NAME.replace("-", "_").replace(":", "_")
    db = chromadb.PersistentClient(path="./chroma_db")

    src_name = f"collection_{safe_embed}_{source_suffix}"
    src = db.get_or_create_collection(src_name)
    if src.count() == 0:
        print(f"❌ Source '{src_name}' is empty! Build it first.")
        return 0
    print(f"📦 [Source] '{src_name}' ({src.count()} chunks)")

    tgt_name = f"collection_{safe_embed}_{target_suffix}"
    try:
        db.delete_collection(tgt_name)
    except Exception:
        pass
    tgt = db.get_or_create_collection(tgt_name)
    print(f"📦 [Target] '{tgt_name}' (fresh)")

    cache_path = "contextual_cache.json"
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"💾 [Cache] {len(cache)} cached headers")
    else:
        cache = {}

    def save_cache():
        with open(cache_path, "w") as f:
            json.dump(cache, f)

    # 🚨 FIX: Τραβάμε και τα αυθεντικά IDs από τη Chroma
    data = src.get(include=["documents", "metadatas"])
    ids = data["ids"]
    texts = data["documents"]
    metas = data["metadatas"]
    print(f"📄 Contextualising {len(texts)} chunks (same count as source!)...")

    storage_context = StorageContext.from_defaults(
        vector_store=ChromaVectorStore(chroma_collection=tgt)
    )

    contextualised = []
    # 🚨 FIX: Κάνουμε zip και το node_id
    for i, (node_id, text, md) in enumerate(zip(ids, texts, metas)):
        md = md or {}
        fname = md.get("file_name", "document")
        key = hashlib.sha256(text.encode()).hexdigest()[:16]

        if key in cache:
            header = cache[key]
        else:
            prompt = (
                "Give a SHORT context (one sentence) situating this chunk "
                "within its document, to improve search retrieval. "
                "State the topic/section. Be concise.\n\n"
                f"Document: {fname}\nChunk:\n{text[:800]}\n\nShort context:"
            )
            header = ""
            for attempt in range(3):
                try:
                    header = str(Settings.llm.complete(prompt)).strip()
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"    ⚠️ Retry {attempt + 1}/3: {e} ({wait}s)")
                    time.sleep(wait)
            cache[key] = header
            time.sleep(0.05)
            if len(cache) % 25 == 0:
                save_cache()

        new_text = f"[Context: {header}]\n{text}"
        # 🚨 FIX: Περνάμε το αυθεντικό node_id στο TextNode
        node = TextNode(text=new_text, metadata=md, id_=node_id)
        contextualised.append(node)

        if (i + 1) % 50 == 0:
            print(f"   ...{i + 1}/{len(texts)}")

    save_cache()
    print(f"🔢 Embedding {len(contextualised)} chunks...")
    VectorStoreIndex(contextualised, storage_context=storage_context)
    print(f"✅ Done! {len(contextualised)} chunks (matches source: {len(contextualised) == src.count()})")
    return len(contextualised)


def classify_files(documents_folder: str = "data"):
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

    return classification, file_hashes


def process_files(documents_folder, classification, file_hashes, decisions: dict):
    print(f"\n{'=' * 60}")
    print("⚙️  [Process] Starting embedding work...")

    coll = _get_collection()
    vector_store = ChromaVectorStore(chroma_collection=coll)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    report = {"added": [], "skipped": list(classification["to_skip"]), "replaced": [], "kept_both": []}

    for fn in classification["to_add"]:
        path = os.path.join(documents_folder, fn)
        _embed_one_file(path, fn, file_hashes[fn], storage_context)
        report["added"].append(fn)

    for fn in classification["conflicts"]:
        choice = decisions.get(fn, "replace")
        path = os.path.join(documents_folder, fn)
        print(f"\n⚖️  [Conflict] '{fn}' -> decision: {choice.upper()}")

        if choice == "replace":
            print(f"🗑️  [Replace] Deleting old chunks of '{fn}'...")
            coll.delete(where={"file_name": fn})
            _embed_one_file(path, fn, file_hashes[fn], storage_context)
            report["replaced"].append(fn)
        else:
            new_name = _make_unique_name(documents_folder, fn)
            new_path = os.path.join(documents_folder, new_name)
            os.rename(path, new_path)
            _embed_one_file(new_path, new_name, file_hashes[fn], storage_context)
            report["kept_both"].append(f"{fn} -> {new_name}")

    return report


def get_hyde_query_engine(index, file_names=None, use_hybrid=True, top_k_retrieve=10, top_n_rerank=3):
    print(f"\n🔮 [HyDE] Building HyDE query engine (hybrid={use_hybrid})")
    if use_hybrid:
        nodes = _load_nodes_from_chroma(file_names=file_names)
        if not nodes:
            return None
        bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k_retrieve)
        filters = _build_filter(file_names)
        vector = index.as_retriever(similarity_top_k=top_k_retrieve, filters=filters)
        retriever = QueryFusionRetriever(
            [vector, bm25], similarity_top_k=top_k_retrieve,
            num_queries=1, mode="reciprocal_rerank", use_async=False,
        )
        reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=top_n_rerank
        )
        postprocessors = [reranker]
    else:
        filters = _build_filter(file_names)
        retriever = index.as_retriever(similarity_top_k=top_n_rerank, filters=filters)
        postprocessors = []

    from llama_index.core.query_engine import RetrieverQueryEngine
    base_engine = RetrieverQueryEngine.from_args(retriever=retriever, node_postprocessors=postprocessors)
    hyde = HyDEQueryTransform(include_original=True)
    hyde_engine = TransformQueryEngine(base_engine, query_transform=hyde)
    return hyde_engine


def load_index():
    coll = _get_collection()
    if coll.count() == 0:
        return None
    vector_store = ChromaVectorStore(chroma_collection=coll)
    return VectorStoreIndex.from_vector_store(vector_store)


def get_chat_engine(index, file_names=None):
    filters = _build_filter(file_names=file_names)
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        similarity_top_k=5,
        filters=filters,
        system_prompt=SYSTEM_PROMPT,
        verbose=False,
    )
    return chat_engine


def _load_nodes_from_chroma(file_names=None):
    coll = _get_collection()
    if coll.count() == 0:
        return []

    data = coll.get(include=["documents", "metadatas"])
    nodes = []
    for node_id, text, md in zip(data["ids"], data["documents"], data["metadatas"]):
        md = md or {}
        if file_names and md.get("file_name") not in file_names:
            continue
        nodes.append(TextNode(text=text, metadata=md, id_=node_id))
    return nodes


def get_hybrid_chat_engine(index, file_names=None, top_k_retrieve: int = 10, top_n_rerank: int = 3):
    scope = file_names if file_names else "ALL documents"
    print(f"\n🚀 [Engine] Building HYBRID chat engine (scope: {scope})")

    nodes = _load_nodes_from_chroma(file_names=file_names)
    if not nodes:
        return None

    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k_retrieve)
    filters = _build_filter(file_names)
    vector_retriever = index.as_retriever(similarity_top_k=top_k_retrieve, filters=filters)
    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever], similarity_top_k=top_k_retrieve,
        num_queries=1, mode="reciprocal_rerank", use_async=False,
    )
    reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=top_n_rerank)

    return CondensePlusContextChatEngine.from_defaults(
        retriever=hybrid_retriever, node_postprocessors=[reranker], system_prompt=SYSTEM_PROMPT, verbose=False
    )


def list_indexed_files():
    coll = _get_collection()
    info = get_indexed_file_info(coll)
    return sorted(info.keys())


def route_question_to_files(question: str, available_files: list):
    if not available_files or len(available_files) == 1:
        return available_files

    file_list = "\n".join(f"{i}. {f}" for i, f in enumerate(available_files))
    prompt = (
        "You are a document router for a search system.\n"
        "Given a question and a list of documents, return the numbers of ALL documents...\n"
        f"Documents:\n{file_list}\n\nQuestion: {question}\n\nNumbers:"
    )
    response = str(Settings.llm.complete(prompt))
    found_numbers = re.findall(r"\d+", response)

    selected = []
    for token in found_numbers:
        idx = int(token)
        if 0 <= idx < len(available_files) and available_files[idx] not in selected:
            selected.append(available_files[idx])
    return selected if selected else available_files


from llama_index.core import QueryBundle


class CRAGQueryEngine:
    def __init__(self, retriever, postprocessors):
        self.retriever = retriever
        self.postprocessors = postprocessors

    def query(self, question: str):
        bundle = QueryBundle(question)
        nodes = self.retriever.retrieve(bundle)
        for pp in self.postprocessors:
            nodes = pp.postprocess_nodes(nodes, query_bundle=bundle)

        grade = grade_chunks(question, nodes)
        if grade == "irrelevant":
            answer = "I couldn't find relevant information to answer that in the provided documents."
        else:
            context = "\n---\n".join(n.text for n in nodes)
            prefix = "" if grade == "relevant" else "Note: the documents only loosely cover this. "
            prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            answer = prefix + str(Settings.llm.complete(prompt))

        return CRAGResponse(answer, nodes)


class CRAGResponse:
    def __init__(self, answer, source_nodes):
        self._answer = answer
        self.source_nodes = source_nodes

    def __str__(self):
        return self._answer