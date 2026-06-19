# build_contextual.py
from dotenv import load_dotenv
from rag_engine import add_contextual_from_existing

load_dotenv()

if __name__ == "__main__":
    print("Building contextual from existing 'sentence' chunks...")
    print("⚠️  Same chunks + header. ~6455 LLM calls (cached/resumable).\n")
    add_contextual_from_existing(
        source_suffix="sentence",
        target_suffix="contextual",
    )