# ab_test_crag.py
"""Compare Hybrid+Rerank baseline vs CRAG (with relevance grading)."""

import warnings
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.metrics import LLMContextPrecisionWithoutReference

from rag_engine import (
    load_index, get_hybrid_chat_engine, get_crag_query_engine,
)

warnings.filterwarnings("ignore")
load_dotenv()

context_precision_nr = LLMContextPrecisionWithoutReference()

TEST_QUESTIONS = [
    ("What is gradient descent?", "ml"),
    ("What is a neural network?", "ml"),
    ("What are macronutrients?", "nutrition"),
    ("What is the role of vitamin D in the body?", "nutrition"),
    ("What is a black hole?", "physics"),
    ("What is the theory of relativity?", "physics"),
("What is the capital of Australia?", "out-of-scope"),
    ("How do I bake a chocolate cake?", "out-of-scope"),
]

def run_chat(engine):
    records = {"question": [], "answer": [], "contexts": [], "qtype": []}
    for q, qt in TEST_QUESTIONS:
        resp = engine.chat(q)
        records["question"].append(q)
        records["answer"].append(str(resp))
        records["contexts"].append([n.text for n in resp.source_nodes])
        records["qtype"].append(qt)
        engine.reset()
    return Dataset.from_dict(records)

def run_query(engine):
    records = {"question": [], "answer": [], "contexts": [], "qtype": []}
    for q, qt in TEST_QUESTIONS:
        resp = engine.query(q)
        records["question"].append(q)
        records["answer"].append(str(resp))
        records["contexts"].append([n.text for n in resp.source_nodes])
        records["qtype"].append(qt)
    return Dataset.from_dict(records)

def score(dataset, qtypes):
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision_nr],
    )
    df = result.to_pandas()
    df["qtype"] = qtypes
    cols = [c for c in df.columns
            if c not in {"user_input", "retrieved_contexts", "response",
                         "reference", "qtype"} and df[c].dtype.kind in "fi"]
    return {m: df[m].mean() for m in cols}, df, cols

def main():
    index = load_index()
    if index is None:
        raise RuntimeError("No index found!")
    qtypes = [q[1] for q in TEST_QUESTIONS]

    print("\n[A] Baseline (Hybrid + Rerank)...")
    base = get_hybrid_chat_engine(index)
    base_s, base_df, cols = score(run_chat(base), qtypes)

    print("\n[B] CRAG (Hybrid + Rerank + grading)...")
    crag = get_crag_query_engine(index, use_hybrid=True)
    crag_s, crag_df, _ = score(run_query(crag), qtypes)

    print("\n" + "=" * 60)
    print("        BASELINE  vs  CRAG")
    print("=" * 60)
    print(f"{'Metric':<42}{'Base':>8}{'CRAG':>8}{'Δ':>9}")
    print("-" * 60)
    for m in base_s:
        a, b = base_s[m], crag_s[m]
        arrow = "↑" if b > a else ("↓" if b < a else "=")
        print(f"{m:<42}{a:>8.3f}{b:>8.3f}{b-a:>+8.3f}{arrow}")
    print("=" * 60)

if __name__ == "__main__":
    main()