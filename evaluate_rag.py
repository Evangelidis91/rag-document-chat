"""Offline evaluation of the RAG pipeline with Ragas 0.2.x."""

from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

from rag_engine import get_chat_engine, get_hybrid_chat_engine, load_index

load_dotenv()

TEST_QUESTIONS = [
    "What is the main topic of the document?",
    "Who is the author?",
    "What conclusion does the document reach?",
    "Top 3 foods to avoid?",
    "top 5 foods to add to my nutrition for gym?"
]


def run_pipeline(use_hybrid: bool = True):
    """Run each test question and collect question / answer / contexts."""
    index = load_index()
    if index is None:
        raise RuntimeError("No index found. Build one in the app first!")

    engine = (
        get_hybrid_chat_engine(index) if use_hybrid else get_chat_engine(index)
    )

    records = {"question": [], "answer": [], "contexts": []}
    for q in TEST_QUESTIONS:
        response = engine.chat(q)
        records["question"].append(q)
        records["answer"].append(str(response))
        records["contexts"].append(
            [node.text for node in response.source_nodes]
        )
        engine.reset()

    return Dataset.from_dict(records)


def main():
    print("Running RAG pipeline on test questions...")
    dataset = run_pipeline(use_hybrid=True)


    # Debug: δες τι ΑΚΡΙΒΩΣ απάντησε το μοντέλο
    for i in range(len(dataset)):
        print(f"\n--- Q{i}: {dataset['question'][i]}")
        print(f"ANSWER: {dataset['answer'][i]}")
        print(f"# contexts: {len(dataset['contexts'][i])}")

    print("Scoring with Ragas...")
    # In 0.2.x, no explicit llm/embeddings needed:
    # Ragas uses OPENAI_API_KEY from the environment by default.
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
    )

    print("\n===== RAGAS RESULTS =====")
    print(result)
    print("\n===== PER-QUESTION =====")
    print(result.to_pandas())


if __name__ == "__main__":
    main()