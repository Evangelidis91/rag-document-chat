# ab_test_chunking.py
"""Compare sentence vs semantic chunking.

Run TWICE: CHUNKING=sentence and CHUNKING=semantic, on the SAME questions.
"""

import warnings
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from ragas.metrics import LLMContextPrecisionWithoutReference

from rag_engine import CHUNKING, get_hybrid_chat_engine, load_index

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


def run(engine):
    records = {"question": [], "answer": [], "contexts": [], "qtype": []}
    for q, qtype in TEST_QUESTIONS:
        resp = engine.chat(q)
        records["question"].append(q)
        records["answer"].append(str(resp))
        records["contexts"].append([n.text for n in resp.source_nodes])
        records["qtype"].append(qtype)
        engine.reset()
    return Dataset.from_dict(records)


def main():
    index = load_index()
    if index is None:
        raise RuntimeError(
            f"No index for CHUNKING={CHUNKING}! Build it first."
        )

    print(f"\n=== Evaluating with CHUNKING={CHUNKING.upper()} ===")
    engine = get_hybrid_chat_engine(index)
    dataset = run(engine)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision_nr],
    )
    df = result.to_pandas()
    df["qtype"] = df.index.map(lambda i: TEST_QUESTIONS[i][1])

    print(f"\n===== RESULTS ({CHUNKING}) =====")
    for col in [
        "faithfulness",
        "answer_relevancy",
        "llm_context_precision_without_reference",
    ]:
        if col in df.columns:
            print(f"{col:<42}: {df[col].mean():.3f}")

    print(f"\nContext precision by domain ({CHUNKING}):")
    cp = "llm_context_precision_without_reference"
    if cp in df.columns:
        for t in df["qtype"].unique():
            mask = df["qtype"] == t
            print(f"  {t:<12}: {df.loc[mask, cp].mean():.3f}")


if __name__ == "__main__":
    main()