# 📄 Chat with Your Documents — A Local RAG Application

A Retrieval-Augmented Generation (RAG) app that lets you chat with your
own documents (PDF, Word, text). Runs **fully local & free** via Ollama,
with an optional OpenAI backend used for benchmarking. Features hybrid
search, reranking, query routing, HyDE, configurable chunking, content
deduplication, and a rigorous Ragas-based evaluation suite.

---

## ✨ Features

- 💬 **Streaming chat** with conversational memory (handles follow-ups)
- 🔍 **Hybrid retrieval** — dense vector search + BM25 keyword search
- 🎯 **Cross-encoder reranking** for higher-quality context
- 🧭 **Smart routing** — automatic (LLM picks documents) or manual
  (multi-select) document selection
- 🔮 **HyDE** (Hypothetical Document Embeddings) query transform
- ✂️ **Configurable chunking** — fixed-size (sentence) or semantic
- 🦙 **Local & free** via Ollama (llama3.2 + nomic-embed-text)
- ☁️ **Optional OpenAI backend** (used for stable benchmarking)
- 🔐 **Content deduplication** via SHA-256 hashing
- ⚖️ **Conflict resolution** — same filename, different content → user chooses
- 📎 **Source citations** with keyword highlighting & page numbers
- 📊 **Stats panel** — documents, chunks, latency, per-document breakdown
- 🧪 **Evaluation suite** (Ragas) with A/B tests & ablation studies
- 🔭 **Observability** — Live trace/span analysis via Arize Phoenix integration
- ✅ **Unit tests** (pytest) for the deduplication logic

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| RAG framework | **LlamaIndex** |
| Vector database | **ChromaDB** (local, persistent) |
| LLM (demo) | **Ollama llama3.2** (local, free) |
| LLM (benchmarks) | **OpenAI gpt-4o-mini** (stable, consistent) |
| Embeddings | **nomic-embed-text** (local) / **text-embedding-3-small** (cloud) |
| Keyword search | **BM25** (`llama-index-retrievers-bm25`) |
| Reranking | **Sentence-Transformers** cross-encoder |
| Query transform | **HyDE** |
| UI | **Streamlit** |
| Evaluation | **Ragas** (pinned to 0.2.x) |
| Observability | **Arize Phoenix** |
| Testing | **pytest** |

---

## 🚀 Setup

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed and running

### Installation

```bash
git clone [https://github.com/Evangelidis91/rag-document-chat.git](https://github.com/Evangelidis91/rag-document-chat.git)
cd rag-document-chat

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Pull the local models (free)
ollama pull llama3.2
ollama pull nomic-embed-text

# (Optional) For benchmarking with OpenAI, add your key
cp .env.example .env   # then set OPENAI_API_KEY & ENABLE_PHOENIX=true
```

### Run (demo, local & free)

```bash
streamlit run app.py
```

Open http://localhost:8501, upload documents, and start chatting.

---

## ⚙️ Configuration

The app is controlled via environment variables:

| Env var | Default | Description |
|---------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` (local) or `openai` (benchmarks) |
| `OLLAMA_MODEL` | `llama3.2` | The local LLM |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | The local embedding model |
| `CHUNKING` | `sentence` | `sentence` (fixed-size) or `semantic` |
| `ENABLE_PHOENIX` | `false` | Set to `true` to launch the Phoenix tracing UI |
| `OPENAI_API_KEY` | — | Required for OpenAI provider & Ragas |

```bash
# Examples
streamlit run app.py                                  # local demo
CHUNKING=semantic streamlit run app.py                # semantic chunking
ENABLE_PHOENIX=true streamlit run app.py              # app with telemetry
LLM_PROVIDER=openai python ab_test.py                 # benchmark
```

---

## 📊 Evaluation & Benchmarks

The project includes a rigorous evaluation suite using **Ragas** (LLM-as-a-judge). Benchmarks use OpenAI gpt-4o-mini for stable, reproducible measurements across a fixed test set spanning 3 domains (Machine Learning, Nutrition, Physics).

> Full results and in-depth empirical analysis in [benchmarks.md](benchmarks.md).

### Retrieval Foundation

| Retrieval        | Faithfulness | Answer Rel. | Context Prec. | Latency |
|------------------|:------------:|:-----------:|:-------------:|:-------:|
| Vector only      |    0.988     |    0.944    |     0.921     |  4.1s   |
| Hybrid + Rerank  |    0.972     |    0.939    |     0.903     |  3.3s   |

### Ablation Study Highlights

| Technique | Key Result | Insight |
|-----------|-----------|---------|
| **HyDE** | +0.072 ctx precision (vector-only); **+0.000 with reranker** | HyDE & reranking are **redundant** — both refine retrieval. Stacking them blindly adds LLM latency for zero gain. |
| **Semantic chunking** | −0.13 faithfulness overall | **Domain-dependent**: helped Nutrition (+0.21) but collapsed ML (−0.50) because mathematical equations/code lack sentence prose structure. |
| **CRAG** | Perfect 1.000 Faithfulness (lenient); Strict mode drops score | Value hinges on the grader. Excellent at **correctly refusing out-of-scope queries**, but prone to flagging complex edge cases as irrelevant. |
| **Contextual Retrieval** | Catastrophic drop across all metrics (Ctx Prec. −0.227) | **"Thematic Swamping Effect"**: Prepending global context to structured textbooks pollutes BM25 IDF weights and creates vector clustering noise. |

---

## 🧪 Testing

```bash
pytest -v
```

Unit tests cover SHA-256 deduplication and file-classification logic, using a fake Chroma collection (no database or LLM required).

---

## 📁 Project Structure

```
rag-app/
├── app.py                 # Streamlit UI (chat, upload, stats, routing)
├── rag_engine.py          # Core RAG logic (parse, embed, retrieve, tracing)
├── build_contextual.py    # Generates contextual chunks utilizing JSON caching
├── ab_test.py             # Vector vs hybrid A/B evaluation
├── ab_test_hyde.py        # HyDE ablation study
├── ab_test_chunking.py    # Sentence vs semantic chunking comparison
├── ab_test_crag.py        # Corrective RAG evaluation & out-of-scope tests
├── ab_test_contextual.py  # Isolated Contextual Retrieval benchmark
├── tests/                 # pytest unit tests
│   └── test_classify.py
├── benchmarks.md          # Full benchmark results & empirical analysis
├── requirements.txt
└── .env.example
```

---

## 🗺️ Roadmap

- [x] Hybrid search + reranking
- [x] Smart routing (auto + manual)
- [x] HyDE query transform
- [x] Configurable chunking (sentence / semantic)
- [x] Streaming responses + source highlighting
- [x] Ablation study framework
- [x] CRAG (Corrective RAG / self-correction)
- [x] Observability (Arize Phoenix trace/span analysis)
- [x] Contextual Retrieval Integration & Evaluation
- [ ] Multimodal support (tables & figures via LlamaParse)

---

## 📝 Notes on Dependencies

Ragas is pinned to `0.2.x`. Newer 0.4.x versions have an unresolved async deadlock with recent `langchain-core` versions. Pinning ensures the evaluation suite runs reliably. See `LESSONS_LEARNED.md` for the full debugging journey.


## Known Limitations & Roadmap

- [x] **PDF hyphenation fix** — joins words split across line breaks (e.g. "macro-\nnutrient") so exact-term search works better.
- [ ] **Advanced parsing** (LlamaParse/Unstructured) for tables & figures.
- [ ] **BM25 index caching** — currently rebuilt per query (slow at scale).
- [ ] **Routing reliability** scales with model size; small local models may occasionally exclude the most relevant document.