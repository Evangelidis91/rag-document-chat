# 📄 Chat with Your Documents — A Local RAG Application

A Retrieval-Augmented Generation (RAG) app that lets you chat with your
own documents (PDF, Word, text). It runs **fully local and free** via
Ollama, with optional OpenAI support. Features hybrid search, cross-encoder
reranking, content-based deduplication, and a Ragas evaluation harness.

---

## ✨ Features

- 💬 **Chat interface** with conversational memory (handles follow-ups)
- 🔍 **Hybrid retrieval** — dense vector search + BM25 keyword search
- 🎯 **Cross-encoder reranking** for higher-quality context
- 🦙 **Local & free** via Ollama (llama3.2 + nomic-embed-text)
- 🔐 **Content deduplication** via SHA-256 hashing (skip identical files)
- ⚖️ **Conflict resolution** — same filename, different content → user chooses
- 📊 **Evaluation harness** (Ragas): faithfulness, answer relevancy and
context precision, plus an A/B test (basic vs hybrid)
- 📎 **Source citations** for every answer
- 🧪 **Unit tests** (pytest) for the deduplication logic
- 🧭 **Hybrid + routing + filtering** — combine semantic & keyword search
with automatic or manual document selection

---

## 🛠️ Tech Stack

| Layer           | Technology                                     |
|-----------------|------------------------------------------------|
| RAG framework   | **LlamaIndex**                                 |
| Vector database | **ChromaDB** (local, persistent)               |
| LLM             | **Ollama llama3.2** (local) — OpenAI optional  |
| Embeddings      | **nomic-embed-text** (local) — OpenAI optional |
| Keyword search  | **BM25** (`llama-index-retrievers-bm25`)       |
| Reranking       | **Sentence-Transformers** cross-encoder        |
| UI              | **Streamlit**                                  |
| Evaluation      | **Ragas** (pinned to 0.2.x)                    |
| Testing         | **pytest**                                     |

---

## 🚀 Setup

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed and running

### Installation

```bash
# 1. Clone & enter the repo
git clone https://github.com/YOUR-USERNAME/rag-document-chat.git
cd rag-document-chat

# 2. Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the Ollama models (local, free)
ollama pull llama3.2
ollama pull nomic-embed-text

# 5. (Optional) For OpenAI / Ragas evaluation, copy and fill .env
cp .env.example .env
```

### Run

```bash
streamlit run app.py
```

Open http://localhost:8501, upload a document, click **Build index**,
and start chatting.

---

## 📊 Evaluation

The app includes an evaluation harness using **Ragas** (LLM-as-a-judge).

Run the A/B test comparing basic vs hybrid retrieval:

```bash
python ab_test.py
```

### Example results (food-science corpus, ~18k chunks, Ollama llama3.2)

| Metric            | Basic | Hybrid + Rerank |    Δ    |
|-------------------|:-----:|:---------------:|:-------:|
| Faithfulness      | 0.858 |      0.857      | −0.002  |
| Answer Relevancy  | 0.945 |      0.944      | −0.001  |
| Context Precision | 0.931 |      1.000      | +0.069  |

**Context precision by question type:**

| Type      | Basic | Hybrid |    Δ     |
|-----------|:-----:|:------:|:--------:|
| Semantic  | 0.861 | 1.000  | +0.139   |
| Keyword   | 1.000 | 1.000  |  0.000   |

**Key findings:**
- Hybrid + reranking improved **context precision**, especially on
semantic queries (+0.139) — the reranker promotes relevant chunks.
- Faithfulness and relevancy were already near-ceiling, so overall gains
were marginal. The value of hybrid search is query-dependent — measured,
not assumed.

---

## 🧪 Testing

```bash
pytest -v
```

Unit tests cover the SHA-256 deduplication and file-classification logic,
using a fake Chroma collection (test double) — no database or LLM needed.

---

## 📁 Project Structure

```
rag-app/
├── app.py              # Streamlit UI (chat, upload, conflict dialog)
├── rag_engine.py       # Core RAG logic (parse, embed, retrieve, chat)
├── ab_test.py          # Basic vs hybrid A/B evaluation
├── evaluate_rag.py     # Single-pipeline Ragas evaluation
├── tests/              # pytest unit tests
├── requirements.txt
└── .env.example
```

---

## ⚙️ Configuration

| Env var              | Default            | Description                        |
|----------------------|--------------------|------------------------------------|
| `OLLAMA_MODEL`       | `llama3.2`         | The local LLM                      |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | The local embedding model          |
| `OPENAI_API_KEY`     | —                  | Required only for Ragas evaluation |

---

## 📝 Notes on Dependencies

Ragas is pinned to `0.2.x`. Newer 0.4.x versions have an unresolved async
deadlock with recent `langchain-core`. Pinning ensures the evaluation
harness runs reliably.

## Known Limitations & Roadmap

- [x] **PDF hyphenation fix** — joins words split across line breaks
    (e.g. "macro-\nnutrient") so exact-term search works better.
- [ ] **Advanced parsing** (LlamaParse/Unstructured) for tables & figures.
- [ ] **BM25 index caching** — currently rebuilt per query (slow at scale).
- [ ] **Routing reliability** scales with model size; small local models
    may occasionally exclude the most relevant document.