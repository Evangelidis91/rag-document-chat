# 🧠 Lessons Learned & Debugging Journey

Real problems encountered while building this RAG application, how they
were diagnosed, and what they taught me. This is the "behind the scenes"
of turning a working demo into a robust, measured system.

---

## 📑 Categories
1. [Dependency & Version Issues](#1-dependency--version-issues)
2. [LlamaIndex API Pitfalls](#2-llamaindex-api-pitfalls)
3. [Streamlit State & Rendering](#3-streamlit-state--rendering)
4. [The "macronutrients" Investigation](#4-the-macronutrients-investigation)
5. [LLM Output Handling](#5-llm-output-handling)
6. [Evaluation Insights (Ablation)](#6-evaluation-insights-ablation)

---

## 1. Dependency & Version Issues

### 1.1 Ragas async deadlock (0.4.x)
**Problem:** `evaluate()` hung indefinitely (stuck at `0/6`), then threw
`CancelledError` / `TimeoutError`. The exception was swallowed silently
by default (`raise_exceptions=False`), masking a failure as a hang.

**Diagnosis:** Set `raise_exceptions=True` to surface the real error —
an async deadlock in langchain-core's callback manager.

**Solution:** Pinned Ragas to `0.2.x` (stable). Documented in requirements.

**Lesson:** In fast-moving AI tooling, the latest version isn't always
best. Pin stable versions for reproducibility. Also: silent exception
handling turns simple bugs into mysterious hangs.

### 1.2 Missing Vertex AI module
**Problem:** `ModuleNotFoundError: langchain_community.chat_models.vertexai`
— even though we only use OpenAI.

**Cause:** Ragas imports it proactively at startup; a class had moved
packages between langchain versions.

**Solution:** Installed `langchain-google-vertexai` to satisfy the import.

**Lesson:** Transitive dependencies can break you with code you never call.

### 1.3 Embedding dimension mismatch (OpenAI vs Ollama)
**Problem:** Switching providers crashed Chroma (1536-dim vs 768-dim
vectors are incompatible).

**Solution:** Provider-specific collection names
(`collection_<embed_model>_<chunking>`).

**Lesson:** Embeddings from different models are NOT interchangeable.
Isolate them by storage namespace.

---

## 2. LlamaIndex API Pitfalls

### 2.1 Wrong fusion mode name
**Problem:** `ValueError: Invalid fusion mode: reciprocal_rank_fusion`.

**Solution:** LlamaIndex uses `"reciprocal_rerank"`, not the generic
academic term "reciprocal rank fusion".

**Lesson:** Library-specific naming often differs from textbook terms.

### 2.2 `Settings.embedding` vs `Settings.embed_model`
**Problem:** Set the wrong attribute → LlamaIndex silently used a default
embedding model instead of the one I chose.

**Lesson:** Wrong-but-valid config fails silently. Verify behaviour, not
just absence of errors.

### 2.3 Document.text is read-only
**Problem:** `AttributeError: property 'text' has no setter` when trying
to clean chunk text (`d.text = ...`).

**Solution:** Build new objects: `Document(text=cleaned, metadata=...)`.

**Lesson:** Pydantic models often expose read-only properties; construct
fresh instances instead of mutating.

### 2.4 BM25 node ID alignment
**Problem:** BM25 (in-memory) and vector (Chroma) retrievers used
different node IDs, so Reciprocal Rank Fusion couldn't deduplicate the
same chunk returned by both.

**Solution:** Reused Chroma's IDs when rebuilding TextNodes for BM25.

**Lesson:** Fusion needs a shared identity to merge results correctly.

---

## 3. Streamlit State & Rendering

### 3.1 The "one-behind" stats counter
**Problem:** After 3 questions, the stats panel showed "2".

**Cause:** Streamlit runs the script top-to-bottom. The sidebar (drawn
first) read the latency list BEFORE the chat handler (lower in the file)
appended the current question.

**Solution:** `st.rerun()` after handling the message.

**Lesson:** Execution ORDER matters. Widgets reflect state as of the
moment they render.

### 3.2 Sources disappeared after rerun
**Problem:** Adding `st.rerun()` made the source citations vanish.

**Cause:** Sources were rendered only inside the `if prompt:` block.
After a rerun (no new prompt), that block didn't run, so they weren't
redrawn from history.

**Solution:** Store sources as serialisable dicts in `session_state`
message history, and re-render them in the history loop.

**Lesson:** Anything that must survive a rerun must live in
`session_state` and be re-rendered from there — not just drawn once
inside an event handler.

### 3.3 Transformers import noise
**Problem:** Dozens of `torchvision` tracebacks flooded the console.

**Cause:** Streamlit's file-watcher scanned every transformers
image-processing module (pulled in by the reranker).

**Solution:** Installed `torchvision` (or disable the watcher via
`.streamlit/config.toml`).

**Lesson:** Distinguish NOISE from ERRORS. The app worked fine; the
watcher was just being verbose.

---

## 4. The "macronutrients" Investigation

The most instructive bug. A query for **"What are macronutrients?"**
returned *"not found"* despite the topic being well covered.

### What we tried (and what failed)

| # | Attempt | Result | Why |
|---|---------|:------:|-----|
| 1 | Inclusive routing | ❌ | Not a routing-scope problem |
| 2 | Hybrid (BM25) | ❌ | The word was hyphenated, so even exact match missed it |
| 3 | Search all docs | ❌ | Problem was in the chunks, not the scope |
| 4 | Increase top_k | ❌ | More chunks, same wrong ones |
| 5 | Rephrase ("carbohydrates, proteins, fats") | ✅ | Concept exists; the exact term was the issue |
| 6 | Strip PDF hyphenation | ✅ | **Root cause fixed** |

### Root Cause
PDF parsing preserved hyphenated line-breaks: **"macronutrient"** was
stored as **"macro-\nnutrient"** (split across lines). Neither dense
embeddings nor BM25 could match the full term.

### Solution
```
python
cleaned = d.get_content().replace("-\n", "").replace("- ", "")
```

### Lessons
1. **Retrieval quality is bounded by ingestion quality** — "garbage in,
   garbage out." No algorithm finds text mangled during parsing.
2. **Systematic elimination beats guessing** — ruling out routing → then
   hybrid → then scope isolated the real cause in the data.
3. **The system prompt worked correctly** — it honestly reported "not
   found" instead of hallucinating.
4. **Query phrasing matters** as much as retrieval strategy.

---

## 5. LLM Output Handling

### 5.1 Router parsing broke on chatty output
**Problem:** The LLM router was asked for comma-separated numbers but
returned *"Based on the question... 0,1,3"*. Naive `split(",")` mangled
it, selecting the wrong documents.

**Solution:** Robust parsing with regex: `re.findall(r"\d+", response)`.

**Lesson:** Never assume an LLM follows output format strictly — parse
defensively, especially with smaller models.

### 5.2 Broken LaTeX in answers
**Problem:** The model returned LaTeX (`\mathbb{R}^n`) that Streamlit's
`write_stream` rendered as raw text.

**Solution:** Added a system-prompt rule: write math in plain text, not
LaTeX.

**Lesson:** Control output format at the prompt level when rendering is
limited.

---

## 6. Evaluation Insights (Ablation)

Measuring each technique — both cumulatively and in isolation — revealed
that "advanced" techniques are NOT universally beneficial.

### 6.1 Hybrid search is domain-dependent
- ML / Physics: context precision **improved** (+0.06 / +0.10)
- Nutrition: context precision **dropped** (−0.21)

**Why:** Nutrition has morphologically-similar terms ("macro-nutrient",
"macro-biotic", "macro-mineral") → BM25 keyword matching caused lexical
false-positives that pure semantic search avoided.

### 6.2 HyDE and reranking are redundant
- Vector-only + HyDE: context precision **+0.072** (ML: **+0.217**)
- Hybrid+Rerank + HyDE: context precision **+0.000**

**Why:** Both HyDE and the cross-encoder reranker refine retrieval. With
a reranker already present, it converges on the same top chunks
regardless of HyDE — so HyDE's retrieval benefit is masked.

**Lesson:** Two techniques solving the same problem don't stack
(1+1≠2). Stacking them blindly adds latency for zero gain.

### 6.3 Semantic chunking collapsed technical content
- Overall: faithfulness **−0.13**
- Nutrition (clean prose): **+0.21** context precision
- ML (equations/code): **−0.50** context precision

**Why:** Semantic chunking splits on per-sentence embedding similarity.
Mathematical notation and code lack clear sentence structure, so it cut
ML content at wrong boundaries. Fixed-size chunking proved more robust.

**Lesson:** Optimal chunking depends on content type. There is no single
"best" strategy.

---

## 🎯 Meta-Lessons

> **1. Measure, don't assume.** Every "improvement" was verified with
> Ragas. Several "best practices" turned out to be neutral or harmful on
> this corpus.
>
> **2. Per-domain analysis beats averages.** Aggregate metrics repeatedly
> hid critical behaviour that only surfaced when broken down by topic.
>
> **3. Distinguish bugs from data-quality issues.** The biggest "bug"
> (macronutrients) was actually a PDF parsing problem, not a code defect.
>
> **4. Read tracebacks carefully.** The Ragas hang, the read-only
> property, the fusion-mode name — each was solved by reading the error,
> not guessing.
>
> **5. Honest negative results are valuable.** Showing that a technique
> *didn't* help (with a clear explanation of why) demonstrates deeper
> understanding than claiming everything works.
