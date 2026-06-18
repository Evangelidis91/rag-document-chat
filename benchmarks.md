# 📊 Performance Benchmarks

Systematic evaluation of RAG techniques using **Ragas** (LLM-as-a-judge).

> **Methodology:** A single fixed test set (6 questions across 3 domains)
> is used across all runs for direct comparability. The RAG LLM is
> **OpenAI gpt-4o-mini** — chosen over the local Ollama model to keep
> generation quality CONSTANT, so we isolate the effect of *retrieval*
> changes rather than LLM variability. Each technique is measured both
> cumulatively and in isolation (ablation).

---

## Test Configuration

- **Documents (6):**
  - ML: Theodoridis *ML From Classics to Deep Networks*, Flux *ML Mathematics in Python*
  - Nutrition: *Advanced Nutrition & Human Metabolism*, Paxton *Naturopathic Nutrition*
  - Physics: Hawking *A Brief History of Time*, Feynman *Six Easy Pieces*
- **Test questions:** 6 (2 per domain)
- **RAG LLM:** OpenAI gpt-4o-mini (temperature 0.1)
- **Embeddings:** text-embedding-3-small
- **Judge (Ragas):** OpenAI
- **Metrics:** Faithfulness, Answer Relevancy, Context Precision (reference-free)

### Test Questions
| Domain | Questions |
|--------|-----------|
| ML | "What is gradient descent?", "What is a neural network?" |
| Nutrition | "What are macronutrients?", "What is the role of vitamin D?" |
| Physics | "What is a black hole?", "What is the theory of relativity?" |

---

## Table 1 — Retrieval Foundation

| Retrieval        | Faithfulness | Answer Rel. | Context Prec. | Latency |
|------------------|:------------:|:-----------:|:-------------:|:-------:|
| Vector only      |    0.988     |    0.944    |     0.921     |  4.1s   |
| Hybrid + Rerank  |    0.972     |    0.939    |     0.903     |  3.3s   |

### Context Precision by Domain
| Domain    | Vector | Hybrid |    Δ     |
|-----------|:------:|:------:|:--------:|
| ML        | 0.944  | 1.000  | +0.056 ↑ |
| Physics   | 0.819  | 0.917  | +0.097 ↑ |
| Nutrition | 1.000  | 0.792  | −0.208 ↓ |

**Finding:** Overall metrics are nearly identical, but the per-domain
breakdown reveals hybrid search is **domain-dependent**. It improves
ML and Physics (distinct technical terms benefit from BM25's exact
matching) but hurts Nutrition: morphologically-similar terms
("macro-nutrient", "macro-biotic", "macro-mineral") cause lexical
false-positives that pure semantic search avoids. A single average
would have hidden this entirely.

---

## Table 2 — HyDE Ablation

HyDE (Hypothetical Document Embeddings) first asks the LLM to write a
hypothetical answer, then searches with that richer text instead of the
short question.

### On Vector-only (clean measurement)
| Configuration | Faithfulness | Answer Rel. | Context Prec. |
|---------------|:------------:|:-----------:|:-------------:|
| Vector only   |    0.951     |    0.942    |     0.928     |
| Vector + HyDE |    0.983     |    0.952    |   **1.000**   |
| **Δ**         |  **+0.033**  |   +0.011    |  **+0.072**   |

### On Hybrid + Rerank (HyDE's effect is masked)
| Configuration       | Faithfulness | Answer Rel. | Context Prec. |
|---------------------|:------------:|:-----------:|:-------------:|
| Hybrid + Rerank     |    0.948     |    0.939    |     0.931     |
| + HyDE              |    0.969     |    0.942    |     0.931     |
| **Δ**               |   +0.022     |   +0.003    |   **0.000**   |

### Context Precision by Domain (Vector + HyDE)
| Domain    | Vector | + HyDE |    Δ     |
|-----------|:------:|:------:|:--------:|
| ML        | 0.783  | 1.000  | +0.217 ↑ |
| Nutrition | 1.000  | 1.000  |   0.000  |
| Physics   | 1.000  | 1.000  |   0.000  |

**Finding:** HyDE significantly improves *vector-only* retrieval (+0.072
context precision, +0.217 on technical ML queries). **But its benefit
vanishes when a reranker is present** (+0.000). HyDE and cross-encoder
reranking are **partially redundant** — both refine retrieval, so the
reranker converges on the same top chunks regardless of HyDE. Stacking
them costs an extra LLM call for zero retrieval gain.

---

## Table 3 — Semantic vs Sentence Chunking

(on Hybrid + Rerank)

| Chunking  | Faithfulness | Answer Rel. | Context Prec. | Chunks |
|-----------|:------------:|:-----------:|:-------------:|:------:|
| Sentence  |    0.954     |    0.939    |     0.903     |  6455  |
| Semantic  |    0.825     |    0.784    |     0.806     |  8896  |
| **Δ**     |  **−0.129**  | **−0.155**  |   **−0.097**  |  +38%  |

### Context Precision by Domain
| Domain    | Sentence | Semantic |    Δ     |
|-----------|:--------:|:--------:|:--------:|
| ML        |  1.000   |  0.500   | −0.500 ↓ |
| Nutrition |  0.792   |  1.000   | +0.208 ↑ |
| Physics   |  0.917   |  0.917   |   0.000  |

**Finding:** Semantic chunking **hurt overall** (−0.13 faithfulness),
but results were domain-dependent. It *improved* Nutrition (clean prose
with clear topic boundaries) but *collapsed* ML (−0.50): mathematical
equations and code lack sentence structure, so per-sentence semantic
splitting cut them at wrong boundaries. It also produced 38% more (and
smaller) chunks, reducing context per chunk. Fixed-size chunking is more
robust for technical/mathematical content.

---

## 🎯 Key Takeaways

1. **No technique is universally better.** Hybrid search, HyDE, and
   semantic chunking each helped some domains and hurt others. Value
   depends on corpus, query type, and content structure.

2. **Techniques can be redundant.** HyDE and reranking both refine
   retrieval — combining them adds latency for no gain (1+1≠2).

3. **Per-domain analysis is essential.** Aggregate metrics repeatedly
   hid critical behaviour (e.g. hybrid's −0.21 on Nutrition, semantic's
   −0.50 on ML) that only appeared when broken down by topic.

4. **Content type drives chunking strategy.** Prose favours semantic
   chunking; equations/code favour fixed-size.

5. **Honest negative results matter.** Two of three "advanced"
   techniques were net-negative on this corpus — measured, not assumed.

---

## Reproducing These Results

```
bash
# Foundation: vector vs hybrid+rerank
LLM_PROVIDER=openai python ab_test.py

# HyDE ablation (edit use_hybrid flag inside)
LLM_PROVIDER=openai python ab_test_hyde.py

# Chunking comparison (run for each strategy)
CHUNKING=sentence LLM_PROVIDER=openai python ab_test_chunking.py
CHUNKING=semantic LLM_PROVIDER=openai python ab_test_chunking.py
```

> Each configuration uses a separate Chroma collection
> (`collection_<embed_model>_<chunking>`), so results never mix.

---

## Pending (Roadmap)

| Technique | Status |
|-----------|--------|
| CRAG (Corrective RAG) | ⏳ Not yet measured |
| Contextual Retrieval | ⏳ Not yet measured |
| Observability (Phoenix) | ⏳ Tooling, not a metric |
```
