# ab_test_contextual.py
"""Compare normal vs contextual retrieval — SAME chunks, HYBRID search."""

import warnings
import chromadb
from datasets import Dataset
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from rag_engine import EMBED_NAME
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from ragas.metrics import LLMContextPrecisionWithoutReference

warnings.filterwarnings("ignore")
load_dotenv()

cp = LLMContextPrecisionWithoutReference()

QUESTIONS = [
    "What is gradient descent?",
    "How does backpropagation work?",
    "What is a loss function?",
    "What is a neural network?",
    "What is supervised learning?",
    "What is overfitting in machine learning?",
    "What are macronutrients?",
    "What is the role of vitamin D in the body?",
    "What are antioxidants?",
    "How does the body metabolise carbohydrates?",
    "What is the function of protein in the body?",
    "What are essential fatty acids?",
    "What is a black hole?",
    "What is the theory of relativity?",
    "What is the Big Bang theory?",
    "What is the atomic theory of matter?",
    "What is conservation of energy?",
    "What is gravitation?",
]


def build_hybrid_engine(suffix, top_k=10, top_n=3):
    """Build a HYBRID (vector + BM25 + rerank) engine for a collection."""
    db = chromadb.PersistentClient(path="./chroma_db")
    safe = EMBED_NAME.replace("-", "_").replace(":", "_")
    coll = db.get_or_create_collection(f"collection_{safe}_{suffix}")
    print(f"📦 {suffix}: {coll.count()} chunks")

    # Vector retriever (from Chroma)
    vs = ChromaVectorStore(chroma_collection=coll)
    index = VectorStoreIndex.from_vector_store(vs)
    vector = index.as_retriever(similarity_top_k=top_k)

    # BM25 retriever (from in-memory nodes)
    data = coll.get(include=["documents", "metadatas"])

    # 🚨 FIX: Αντί για id_=str(i), περνάμε το node_id απεριόριστα από το data["ids"]
    nodes = [
        TextNode(text=t, metadata=m or {}, id_=node_id)  # ← Χρήση του αυθεντικού Chroma ID
        for node_id, t, m in zip(data["ids"], data["documents"], data["metadatas"])
    ]
    bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)

    # Fuse + rerank
    fusion = QueryFusionRetriever(
        [vector, bm25],
        similarity_top_k=top_k,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=top_n
    )
    return RetrieverQueryEngine.from_args(
        retriever=fusion, node_postprocessors=[reranker]
    )


def run(engine):
    rec = {"question": [], "answer": [], "contexts": []}
    for q in QUESTIONS:
        resp = engine.query(q)
        rec["question"].append(q)
        rec["answer"].append(str(resp))
        rec["contexts"].append([n.text for n in resp.source_nodes])
    return Dataset.from_dict(rec)


def score(ds):
    r = evaluate(ds, metrics=[faithfulness, answer_relevancy, cp])
    df = r.to_pandas()
    return {
        c: df[c].mean() for c in df.columns if df[c].dtype.kind in "fi"
    }


def main():
    print(f"\nEvaluating {len(QUESTIONS)} questions (HYBRID search)...")

    print("\n[A] Normal (hybrid)...")
    normal = score(run(build_hybrid_engine("sentence")))

    print("\n[B] Contextual (hybrid)...")
    ctx = score(run(build_hybrid_engine("contextual")))

    print("\n" + "=" * 55)
    print(f"    NORMAL vs CONTEXTUAL (hybrid, {len(QUESTIONS)} Q)")
    print("=" * 55)
    for m in normal:
        a, b = normal[m], ctx[m]
        arrow = "↑" if b > a else ("↓" if b < a else "=")
        print(f"{m:<42}{a:>6.3f}{b:>7.3f} {b - a:+.3f}{arrow}")


if __name__ == "__main__":
    main()