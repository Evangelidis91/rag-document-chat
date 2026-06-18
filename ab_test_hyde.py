# ab_test_hyde.py
"""A/B test: Hybrid+Rerank baseline vs Hybrid+Rerank+HyDE.

Measures faithfulness, answer_relevancy, context precision (reference-free)
and breaks down context precision results per domain.
"""

import warnings
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from ragas.metrics import LLMContextPrecisionWithoutReference

from rag_engine import (
    get_hyde_query_engine,
    get_hybrid_chat_engine,
    load_index,
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
]


def run_chat(engine):
    """Run questions through a CHAT engine (baseline)."""
    records = {"question": [], "answer": [], "contexts": [], "qtype": []}
    for q, qtype in TEST_QUESTIONS:
        resp = engine.chat(q)
        records["question"].append(q)
        records["answer"].append(str(resp))
        records["contexts"].append([n.text for n in resp.source_nodes])
        records["qtype"].append(qtype)
        engine.reset()
    return Dataset.from_dict(records)


def run_query(engine):
    """Run questions through a QUERY engine (HyDE)."""
    records = {"question": [], "answer": [], "contexts": [], "qtype": []}
    for q, qtype in TEST_QUESTIONS:
        resp = engine.query(q)
        records["question"].append(q)
        records["answer"].append(str(resp))
        records["contexts"].append([n.text for n in resp.source_nodes])
        records["qtype"].append(qtype)

        # DEBUG: ποια chunks ήρθαν;
        print(f"\n🔍 Q: {q}")
        for n in resp.source_nodes[:2]:
            fname = n.metadata.get("file_name", "?")[:30]
            print(f"   → {fname}: {n.text[:60]}...")
    return Dataset.from_dict(records)


def score(dataset, qtypes):
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision_nr],
    )
    df = result.to_pandas()
    df["qtype"] = qtypes
    metric_cols = [
        c
        for c in df.columns
        if c
        not in {
            "user_input",
            "retrieved_contexts",
            "response",
            "reference",
            "qtype",
        }
        and df[c].dtype.kind in "fi"
    ]
    return {m: df[m].mean() for m in metric_cols}, df, metric_cols


def main():
    index = load_index()
    if index is None:
        raise RuntimeError("No index found!")

    qtypes = [q[1] for q in TEST_QUESTIONS]

    # Baseline: VECTOR ONLY (no rerank, no HyDE) — clean comparison
    print("\n[A] Baseline (Vector only)...")
    from rag_engine import get_chat_engine
    base = get_chat_engine(index)  # basic vector chat engine
    base_scores, base_df, cols = score(run_chat(base), qtypes)

    # HyDE: VECTOR ONLY + HyDE
    print("\n[B] Vector + HyDE...")
    hyde = get_hyde_query_engine(index, use_hybrid=False)  # ← False!
    hyde_scores, hyde_df, _ = score(run_query(hyde), qtypes)

    # Comparison
    print("\n" + "=" * 60)
    print("        BASELINE  vs  + HyDE")
    print("=" * 60)
    print(f"{'Metric':<42}{'Base':>8}{'HyDE':>8}{'Δ':>9}")
    print("-" * 60)
    for m in base_scores:
        a, b = base_scores[m], hyde_scores[m]
        arrow = "↑" if b > a else ("↓" if b < a else "=")
        print(f"{m:<42}{a:>8.3f}{b:>8.3f}{b-a:>+8.3f}{arrow}")
    print("=" * 60)

    # Per-domain context precision (the key metric!)
    cp = "llm_context_precision_without_reference"
    if cp in cols:
        print("\nCONTEXT PRECISION BY DOMAIN:")
        for t in base_df["qtype"].unique():
            mask = base_df["qtype"] == t
            a = base_df.loc[mask, cp].mean()
            b = hyde_df.loc[mask, cp].mean()
            arrow = "↑" if b > a else ("↓" if b < a else "=")
            print(f"  {t:<12}{a:>7.3f}{b:>7.3f}  {b-a:+.3f}{arrow}")


if __name__ == "__main__":
    main()