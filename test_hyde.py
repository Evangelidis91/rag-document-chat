# test_hyde.py
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from rag_engine import load_index   # φορτώνει τα Settings (OpenAI)

hyde = HyDEQueryTransform(include_original=True)
query_bundle = hyde.run("What are macronutrients?")

print("Original:", query_bundle.query_str)
print("\nEmbedding strings (what it searches with):")
for s in query_bundle.embedding_strs:
    print("---")
    print(s[:300])
