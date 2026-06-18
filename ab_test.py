"""A/B test: BASIC (vector only) vs HYBRID (hybrid + rerank) using Ragas.

Measures faithfulness, answer_relevancy, context precision (reference-free)
and per-query latency, then breaks results down per question and by type.
"""

import time
import warnings
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from ragas.metrics import LLMContextPrecisionWithoutReference

from rag_engine import get_chat_engine, get_hybrid_chat_engine, load_index

warnings.filterwarnings("ignore")

load_dotenv()

# Each question is tagged with its type: "semantic" or "keyword"
TEST_QUESTIONS = [
    # Machine Learning
    ("What is gradient descent?", "ml"),
    ("What is a neural network?", "ml"),
    # Nutrition
    ("What are macronutrients?", "nutrition"),
    ("What is the role of vitamin D in the body?", "nutrition"),
    # Physics
    ("What is a black hole?", "physics"),
    ("What is the theory of relativity?", "physics"),
]


def run_pipeline(engine):
    """Run all test questions and collect data + latency + qtype.

    :param engine: A chat engine (basic or hybrid).
    :return: (Dataset for Ragas, list of latencies, list of qtypes).
    """
    records = {"question": [], "answer": [], "contexts": []}
    latencies = []
    qtypes = []

    for question, qtype in TEST_QUESTIONS:
        start = time.time()
        response = engine.chat(question)
        elapsed = time.time() - start

        records["question"].append(question)
        records["answer"].append(str(response))
        records["contexts"].append(
            [node.text for node in response.source_nodes]
        )
        latencies.append(elapsed)
        qtypes.append(qtype)
        engine.reset()

    dataset = Dataset.from_dict(records)
    return dataset, latencies, qtypes

def detect_metric_columns(df):
    """Find which metric columns Ragas actually produced (names vary between
    versions). Returns the numeric metric columns only.

    :param df: The DataFrame from result.to_pandas().
    :return: A list of metric column names.
    """
    # Columns we added ourselves or that Ragas adds for the inputs
    non_metric = {
        "user_input",
        "retrieved_contexts",
        "response",
        "reference",
        "qtype",
        "latency",
        "question",
        "answer",
        "contexts",
    }
    return [
        c
        for c in df.columns
        if c not in non_metric and df[c].dtype.kind in "fi"  # float / int
    ]

# Create an instance of the reference-free context precision metric
context_precision_nr = LLMContextPrecisionWithoutReference()

def score(dataset, qtypes):
    """Run Ragas, then re-attach qtype to the result DataFrame.

    :param dataset: The dataset produced by run_pipeline().
    :param qtypes: The list of question types (semantic/keyword).
    :return: (scores_dict, dataframe, metric_cols).
    """
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision_nr],
    )
    df = result.to_pandas()

    # Ragas rebuilds the df and drops our custom columns -> re-attach qtype
    df["qtype"] = qtypes

    metric_cols = detect_metric_columns(df)
    scores = {m: df[m].mean() for m in metric_cols}
    return scores, df, metric_cols

def detailed_comparison(basic_df, hybrid_df, metric_cols):
    """Print a per-question comparison so we can see WHERE hybrid helps or
    hurts (the average hides this).

    :param basic_df: to_pandas() result from the basic run.
    :param hybrid_df: to_pandas() result from the hybrid run.
    :param metric_cols: The list of metric column names to show.
    :return: None
    """
    print("\n" + "=" * 75)
    print("            PER-QUESTION BREAKDOWN")
    print("=" * 75)
    for i in range(len(basic_df)):
        q = basic_df["user_input"][i][:50]
        print(f"\nQ{i + 1}: {q}...")
        for m in metric_cols:
            a = basic_df[m][i]
            b = hybrid_df[m][i]
            d = b - a
            print(f"   {m:<42} basic={a:.2f}  hybrid={b:.2f}  Δ={d:+.2f}")


def summary_by_type(basic_df, hybrid_df, metric_cols):
    """Group each metric by question type (semantic vs keyword) to reveal WHERE
    hybrid retrieval actually helps.

    :param basic_df: to_pandas() result from the basic run.
    :param hybrid_df: to_pandas() result from the hybrid run.
    :param metric_cols: The list of metric column names to summarise.
    :return: None
    """
    types = basic_df["qtype"].unique()

    for m in metric_cols:
        print("\n" + "=" * 60)
        print(f"     {m.upper()}  BY QUESTION TYPE")
        print("=" * 60)
        print(f"{'Type':<14}{'Basic':>10}{'Hybrid':>10}{'Δ':>12}")
        print("-" * 60)
        for t in types:
            mask = basic_df["qtype"] == t
            a = basic_df.loc[mask, m].mean()
            b = hybrid_df.loc[mask, m].mean()
            delta = b - a
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
            print(f"{t:<14}{a:>10.3f}{b:>10.3f}{delta:>+11.3f}{arrow}")
        print("=" * 60)


def main():
    index = load_index()
    if index is None:
        raise RuntimeError("No index found. Build one in the app first!")

    # --- Configuration A: BASIC ---
    print("\n[A] Running BASIC pipeline (vector only)...")
    basic_engine = get_chat_engine(index)
    basic_dataset, basic_lat, qtypes = run_pipeline(basic_engine)
    print("[A] Scoring with Ragas...")
    basic_scores, basic_df, metric_cols = score(basic_dataset, qtypes)

    # --- Configuration B: HYBRID ---
    print("\n[B] Running HYBRID pipeline (hybrid + rerank)...")
    hybrid_engine = get_hybrid_chat_engine(index)
    hybrid_dataset, hybrid_lat, _ = run_pipeline(hybrid_engine)
    print("[B] Scoring with Ragas...")
    hybrid_scores, hybrid_df, _ = score(hybrid_dataset, qtypes)

    # --- Overall comparison table ---
    print("\n" + "=" * 70)
    print("                        A/B TEST RESULTS")
    print("=" * 70)
    print(f"{'Metric':<42}{'Basic':>9}{'Hybrid':>9}{'Δ':>10}")
    print("-" * 70)
    for metric in basic_scores:
        a = basic_scores[metric]
        b = hybrid_scores[metric]
        delta = b - a
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"{metric:<42}{a:>9.3f}{b:>9.3f}{delta:>+9.3f}{arrow}")
    print("=" * 70)

    # --- Latency (now from the separate lists) ---
    avg_basic = sum(basic_lat) / len(basic_lat)
    avg_hybrid = sum(hybrid_lat) / len(hybrid_lat)
    print(
        f"\nAvg latency/query   Basic: {avg_basic:.2f}s   "
        f"Hybrid: {avg_hybrid:.2f}s   (+{avg_hybrid - avg_basic:.2f}s)"
    )

    # --- Deeper analysis ---
    detailed_comparison(basic_df, hybrid_df, metric_cols)
    summary_by_type(basic_df, hybrid_df, metric_cols)


if __name__ == "__main__":
    main()