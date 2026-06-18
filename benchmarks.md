# 📊 Performance Benchmarks

All measurements use **Ragas** (LLM-as-a-judge) over a fixed set of 6
questions. The RAG LLM is **OpenAI gpt-4o-mini** (chosen for stable,
consistent benchmarking); the demo app runs locally on Ollama.

> **Methodology:** A single fixed test set is used across all runs so
> results are directly comparable. Each technique is measured both
> cumulatively (Table 2) and in isolation (Table 3, ablation study).

---

## Table 1 — Retrieval Foundation

| Retrieval        | Faithfulness | Answer Rel. | Context Prec. | Latency |
|------------------|:------------:|:-----------:|:-------------:|:-------:|
| Vector only      |    0.988     |    0.944    |     0.921     |  4.13s  |
| Hybrid + Rerank  |    0.972     |    0.939    |     0.903     |  3.32s  |

### Context Precision by domain (the real story)

| Domain    | Vector | Hybrid |    Δ     |
|-----------|:------:|:------:|:--------:|
| ML        | 0.944  | 1.000  | +0.056 ↑ |
| Physics   | 0.819  | 0.917  | +0.097 ↑ |
| Nutrition | 1.000  | 0.792  | −0.208 ↓ |

**Finding:** Overall metrics are near-identical, but the per-domain
breakdown reveals the real behaviour: hybrid search improves context
precision on ML (+0.06) and Physics (+0.10), where technical terms are
distinct, but hurts Nutrition (−0.21), where morphologically-similar

---

## Table 2 — Cumulative Enhancements (on Hybrid + Rerank)

| Configuration         | Faithfulness | Answer Rel. | Context Prec. | Latency |
|-----------------------|:------------:|:-----------:|:-------------:|:-------:|
| Baseline (h+rerank)   |      —       |      —      |       —       |    —    |
| + HyDE                |      —       |      —      |       —       |    —    |
| + Semantic chunking   |      —       |      —      |       —       |    —    |
| + CRAG                |      —       |      —      |       —       |    —    |

---

## Table 3 — Ablation Study (each in isolation)

| Configuration         | Faithfulness | Answer Rel. | Context Prec. | Latency |
|-----------------------|:------------:|:-----------:|:-------------:|:-------:|
| Baseline (h+rerank)   |      —       |      —      |       —       |    —    |
| Baseline + HyDE only  |      —       |      —      |       —       |    —    |
| Baseline + Semantic   |      —       |      —      |       —       |    —    |
| Baseline + CRAG only  |      —       |      —      |       —       |    —    |
| **All combined**      |      —       |      —      |       —       |    —    |

---

## Test Configuration

- **Documents (6):** Theodoridis (ML), Flux (ML), Advanced Nutrition,
  Paxton (Nutrition), Hawking (Physics), Feynman (Physics)
- **Test questions:** 6 (2 per topic)
- **RAG LLM:** OpenAI gpt-4o-mini
- **Embeddings:** text-embedding-3-small
- **Judge (Ragas):** OpenAI
- **Date:** _(σήμερα)_

## Key Findings
_(filled in as results come in)_
```
`

---

## 🔧 Βήμα 3: Ενημέρωσε το `ab_test.py` με τις 6 ερωτήσεις
```python
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
```
> ⚠️ Σιγουρέψου ότι αυτές οι ερωτήσεις **όντως** απαντιούνται στα βιβλία σου! Αν κάποια όχι, άλλαξέ την.

---

## 📏 Βήμα 4: Τρέξε το Table 1 (Foundation)

Το `ab_test.py` συγκρίνει ήδη **basic vs hybrid**. Με `LLM_PROVIDER=openai`:
```bash
LLM_PROVIDER=openai python ab_test.py
```
Αυτό σου δίνει **2 από τις 3** γραμμές του Table 1:
- **Vector only** (basic)
- **Hybrid+Rerank** (το hybrid σου ήδη έχει rerank)

> 💡 Το «hybrid χωρίς rerank» (μεσαία γραμμή) είναι extra — αν θες και τις 3, το προσθέτουμε. Αλλιώς, **vector vs hybrid+rerank** είναι αρκετό για το Table 1!

---

## ⚠️ Σημαντικό: re-index με OpenAI πρώτα!

Επειδή αλλάζεις σε OpenAI embeddings (διαφορετική collection):
```bash
# 1. Καθάρισε
rm -rf chroma_db/ data/

# 2. Build με OpenAI
LLM_PROVIDER=openai streamlit run app.py
# → ανέβασε τα 6 βιβλία → Build (γρήγορο με OpenAI!)
# → σταμάτα (Ctrl+C)

# 3. Τρέξε benchmark
LLM_PROVIDER=openai python ab_test.py
```
---

## 🎯 Σύνοψη βημάτων
```
1. ✅ Commit + merge το feature/benchmarks-setup
2. 📊 Φτιάξε benchmarks.md (3 tables)
3. 🔧 Βάλε τις 6 ερωτήσεις στο ab_test.py
4. 🗑️ rm -rf chroma_db/ data/
5. 📚 Build 6 βιβλία με LLM_PROVIDER=openai
6. 📏 LLM_PROVIDER=openai python ab_test.py → Table 1
7. ✍️ Κατέγραψε τα νούμερα στο benchmarks.md
```
---

## 🤔 Πριν προχωρήσεις — επιβεβαίωσε

1. ✅ Έκανες **commit + merge** το branch;
2. 📚 Έχεις τα **6 βιβλία** έτοιμα (όλα PDF τώρα);
3. 🔍 Οι 6 ερωτήσεις απαντιούνται στα βιβλία σου;

> 💡 Πρότασή μου: κάνε **πρώτα** το commit/merge (βήμα 1), μετά το re-index με OpenAI (βήματα 4-5), και πες μου όταν είσαι έτοιμος για το **πρώτο benchmark run** (βήμα 6). Θα ερμηνεύσουμε μαζί τα αποτελέσματα του Table 1!

Πού είσαι τώρα; Έτοιμος για commit, ή έχεις ερώτηση; 🚀😊