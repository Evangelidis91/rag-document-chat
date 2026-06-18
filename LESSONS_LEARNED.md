# 🧠 Lessons Learned & Debugging Journey

Real problems encountered while building this RAG app, and how they were
diagnosed and solved.

---

## 1. Ragas version hell (0.4.x async deadlock)
**Problem:** `evaluate()` hung indefinitely with Ragas 0.4.3 due to an
async deadlock with langchain-core.
**Solution:** Pinned Ragas to 0.2.x. Documented in requirements.txt.
**Lesson:** In fast-moving AI tooling, the latest version isn't always
the best — pin stable versions for reproducibility.

---

## 2. Wrong fusion mode name
**Problem:** `ValueError: Invalid fusion mode: reciprocal_rank_fusion`.
**Solution:** LlamaIndex uses `"reciprocal_rerank"`, not the generic term.
**Lesson:** Library-specific naming differs from general terminology.

---

## 3. Embedding dimension mismatch (OpenAI vs Ollama)
**Problem:** Switching providers crashed Chroma (1536 vs 768 dims).
**Solution:** Provider/model-specific collection names.
**Lesson:** Embeddings from different models are not interchangeable.

---

## 4. Retrieval failure: "macronutrients" not found
**Problem:** Querying "macronutrients" returned macrobiotic/macromineral
chunks; the system said "not found".
**Diagnosis path:**
- Ruled out routing (failed even searching all documents).
- Ruled out hybrid vs basic (both failed).
- Root cause: PDF hyphenation split words ("macro-\nnutrient"), so
neither BM25 nor embeddings matched. Also, the exact term may not
appear verbatim — querying "carbohydrates, proteins, fats" worked.
**Solution:** Strip hyphenated line-breaks during ingestion.
**Lesson:** Retrieval quality is bounded by ingestion quality —
"garbage in, garbage out". The system prompt correctly prevented
hallucination by honestly reporting "not found".

---

## 5. LLM router output parsing
**Problem:** Small model (llama3.2) wrapped its answer in chatty text
("Based on the question... 0,1,3"), breaking naive comma-splitting.
**Solution:** Robust parsing with regex (`re.findall(r"\d+", ...)`).
**Lesson:** Never assume an LLM follows output format strictly — parse
defensively.



## What We Tried — Diagnosing the "macronutrients" Retrieval Failure

A query for **"What are macronutrients?"** returned *"not found"*, even
though the documents clearly cover the topic. Here is the full debugging
journey — including everything that **didn't** work:

| # | What we tried                                                   | Result  | Why                                                                            |
|---|-----------------------------------------------------------------|:-------:|--------------------------------------------------------------------------------|
| 1 | **Inclusive routing** (router selects more documents)           |    ❌    | The problem wasn't routing scope                                               |
| 2 | **Hybrid search** (BM25 + vector)                               |    ❌    | The word was hyphenated, so even BM25's exact match missed it                  |
| 3 | **Routing OFF** (search all documents)                          |    ❌    | The issue was in the chunks themselves, not the scope                          |
| 4 | **Increase top_k** (3 → 5)                                      |    ❌    | More chunks, but still the wrong ones (macrobiotic, not macronutrient)         |
| 5 | **Rephrase the query** ("carbohydrates, proteins, fats")        |    ✅    | Proved the *concept* exists and is findable — the *exact term* was the problem |
| 6 | **PDF hyphenation fix** (strip "-\n" and "- " during ingestion) |    ✅    | **Root cause solved!** "macro-\nnutrient" → "macronutrient"                    |

### Root Cause
PDF parsing preserved hyphenated line-breaks: the word **"macronutrient"**
was stored as **"macro-\nnutrient"** (split across lines). Neither dense
embeddings nor BM25 keyword search could match the full term against the
broken text.

### Solution
Strip hyphenated line-breaks during document ingestion:
```python
cleaned = d.get_content().replace("-\n", "").replace("- ", "")
```

### Key Lessons
1. **Retrieval quality is bounded by ingestion quality** — "garbage in,
garbage out". No retrieval algorithm can find text that was mangled
during parsing.
2. **Systematic elimination beats guessing** — by ruling out routing,
then hybrid, then scope, we isolated the real cause in the data.
3. **The system prompt worked correctly** — it honestly reported
"not found" instead of hallucinating an answer.
4. **Query phrasing matters** — the same concept failed as "macronutrients"
but succeeded as "carbohydrates, proteins, fats".


🎓 Το ΤΕΡΑΣΤΙΟ insight (LinkedIn gold!)
«Hybrid search isn't universally better — its value is domain-dependent. On ML and Physics queries, BM25's exact-term matching improved context precision (+0.06, +0.10). But on Nutrition, it HURT (-0.21): morphologically-similar terms (macro-nutrient, macro-biotic, macro-mineral) caused lexical false-positives that pure semantic search avoided. This shows why per-domain evaluation matters — a single average would have hidden this completely.»

🎓 Το μάθημα που ήδη ανακαλύψαμε (gold!)

    «HyDE improved faithfulness (+0.05) but didn't change context precision when combined with a reranker. Why? The cross-encoder reranker re-scores all candidates regardless of how they were retrieved, converging on the same top chunks whether or not HyDE was applied. HyDE's retrieval benefit is masked by strong reranking — they're partially redundant. This is exactly why ablation studies matter: a technique's value depends on what else is in the pipeline.»


🌟 Το LinkedIn story που χτίζεται

    «Ran an ablation study on HyDE. Surprising finding: on vector-only retrieval, HyDE boosted context precision by +0.07 (and +0.22 on technical ML queries!). But when added on top of a reranker, the gain disappeared — HyDE and cross-encoder reranking are partially redundant, both refining retrieval. The lesson: technique value is contextual. Stacking 'best practices' blindly can mean paying extra latency for zero gain.»

    🎯 Αυτό δείχνει βαθιά κατανόηση — ότι οι τεχνικές δεν είναι additive, μπορεί να επικαλύπτονται. Senior-level insight!
