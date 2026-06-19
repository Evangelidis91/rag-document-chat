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
| Vector + HyDE |    0.983     |    0.952    |   **1.000** |
| **Δ** |  **+0.033** |   +0.011    |  **+0.072** |

### On Hybrid + Rerank (HyDE's effect is masked)
| Configuration       | Faithfulness | Answer Rel. | Context Prec. |
|---------------------|:------------:|:-----------:|:-------------:|
| Hybrid + Rerank     |    0.948     |    0.939    |     0.931     |
| + HyDE              |    0.969     |    0.942    |     0.931     |
| **Δ** |   +0.022     |   +0.003    |   **0.000** |

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
| **Δ** |  **−0.129** | **−0.155** |   **−0.097** |  +38%  |

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

## Table 4 — CRAG (Corrective RAG)

CRAG adds a relevance-grading step: before answering, the LLM judges
whether the retrieved context actually addresses the question, and
refuses to answer if it doesn't.

| Configuration       | Faith. | Ans.Rel | Ctx.Prec | Latency |
|---------------------|:------:|:-------:|:--------:|:-------:|
| Hybrid+Rerank       | 0.941  |  0.939  |  0.903   |  ~3s    |
| + CRAG (strict)     | 0.833  |  0.778  |  0.667   |  ~6s    |
| + CRAG (lenient)    | 1.000  |  0.936  |  0.833   |  ~6s    |

### Out-of-scope questions (the real test)
Two off-topic questions ("capital of Australia?", "chocolate cake
recipe?") that the documents don't cover:

| Configuration   | Behaviour                                            |
|-----------------|------------------------------------------------------|
| Hybrid + Rerank | May attempt an answer from loosely-matched chunks    |
| + CRAG          | Grader returns **"irrelevant"** → correctly refuses  |

**Finding:** CRAG's value hinges on its grader.
- A **strict grader backfired** (−0.13 faithfulness): it wrongly flagged
  valid content as irrelevant (even rejecting Hawking's black-hole
  passage), refusing answers it actually had.
- A **lenient grader** recovered and improved in-scope faithfulness to a
  perfect 1.000 — but added an LLM call per query (~2× latency).
- On **out-of-scope** queries, CRAG correctly identified irrelevant
  retrieval and refused to answer — its true strength.

**Takeaway:** The grader is a double-edged sword. On clean, in-scope
retrieval CRAG adds latency for marginal gain; its real value is
**rejecting out-of-scope queries** and filtering noisy retrieval.

---

## Table 5 — Contextual Retrieval

Contextual Retrieval prepends an LLM-generated summary to each chunk to prevent loss of context during document splitting. To isolate the contextual variable correctly, this run reuses the exact same 6,455 sentence chunks, injecting a shared vector/keyword identity via matched node IDs.

(on Hybrid + Rerank)

| Configuration        | Faithfulness | Answer Rel. | Context Prec. | Chunks |
|----------------------|:------------:|:-----------:|:-------------:|:------:|
| Normal (Hybrid)      |    0.946     |    0.951    |     0.880     |  6455  |
| Contextual (Hybrid)  |    0.798     |    0.892    |     0.653     |  6455  |
| **Δ**                |  **−0.148**  | **−0.059**  |  **−0.227**   | **0**  |

**Finding:** Contextual Retrieval **severely degraded performance across all metrics**, with a catastrophic drop in Context Precision (−0.227) and Faithfulness (−0.148). This negative result reveals a critical structural limitation called the **"Thematic Swamping Effect"**, which occurs when applying this technique to continuous, homogeneous corpora like books:

1. **IDF Pollution in BM25:** Because the 6 books have internal thematic continuity, the LLM generated highly repetitive context prefixes (e.g., *"This chunk is from a book on machine learning and optimization..."*). Words like "machine", "learning", and "chapter" became hyper-frequent, completely polluting the Inverse Document Frequency (IDF) weights of the BM25 retriever and blinding lexical matching.
2. **Vector Space Compression:** Prepending repetitive prefixes diluted the unique semantic vector of each node (`text-embedding-3-small`). It compressed all chunks belonging to the same book into tight, indistinguishable clusters. For granular questions, the retriever pulled adjacent but uninformative chunks, ruining precision.
3. **Extrapolation Hallucinations:** Because Context Precision tanked, the generation LLM (`gpt-4o-mini`) received instructionally void fragments that matched the *global topic* but missed the *exact answer*. Forced by the system prompt to synthesize a response, the LLM relied on outside knowledge or extrapolation, dragging down Faithfulness to 0.798.

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
   
6. **Self-correction needs a reliable grader.** CRAG's relevance grader
   can become a failure point — a strict version rejected valid content.
   Its real value is rejecting out-of-scope queries, not refining clean
   retrieval.

7. **Corpus Topology dictates Contextual Retrieval success.** Anthropic’s baseline benchmarks showed +35% improvement because they used large sets of *independent, fragmented documents* (e.g., separate legal contracts, distinct support tickets). In an *already homogeneous corpus* (continuous textbook prose), adding global headers backfires completely, acting as semantic noise and blinding the hybrid retriever.

---

## Reproducing These Results

```bash
# Foundation: vector vs hybrid+rerank
LLM_PROVIDER=openai python ab_test.py

# HyDE ablation (edit use_hybrid flag inside)
LLM_PROVIDER=openai python ab_test_hyde.py

# Chunking comparison (run for each strategy)
CHUNKING=sentence LLM_PROVIDER=openai python ab_test_chunking.py
CHUNKING=semantic LLM_PROVIDER=openai python ab_test_chunking.py

# Contextual Retrieval A/B test (requires cached json or clean run)
LLM_PROVIDER=openai python ab_test_contextual.py