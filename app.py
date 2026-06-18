import os
import re
import time
import streamlit as st
from rag_engine import (
    classify_files,
    get_chat_engine,
    get_hybrid_chat_engine,
    get_index_stats,
    list_indexed_files,
    load_index,
    process_files,
    route_question_to_files,
)

# Common words to ignore when highlighting
_STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "to", "in", "on", "and",
    "or", "for", "how", "why", "when", "who", "which", "that", "this",
    "with", "do", "does", "can", "explain", "tell", "me", "about",
}


def highlight_text(text: str, query: str, max_len: int = 400) -> str:
    """Highlight the query's keywords inside a chunk (marker-pen style).

    :param text: The chunk text.
    :param query: The user's question.
    :param max_len: Max characters to show.
    :return: An HTML string with keywords highlighted.
    """
    snippet = text[:max_len].strip()
    if len(text) > max_len:
        snippet += "..."

    keywords = [
        w
        for w in re.findall(r"\w+", query.lower())
        if w not in _STOPWORDS and len(w) > 2
    ]
    for kw in keywords:
        pattern = re.compile(rf"\b({re.escape(kw)})\b", re.IGNORECASE)
        snippet = pattern.sub(
            r'<span style="background-color: #FF2C2C; '
            r'font-weight: bold; text-decoration: underline;">\1</span>',
            snippet,
        )
    return snippet


def render_sources(source_data, query=""):
    """Render a list of sources inside an expander. Reused both for fresh

    answers and when redrawing history after a rerun.

    :param source_data: List of dicts with file_name, page, score, text.
    :param query: The question (for keyword highlighting).
    :return: None
    """
    if not source_data:
        return

    with st.expander(f"📎 Sources ({len(source_data)})"):
        for i, src in enumerate(source_data, 1):
            relevance = max(0, min(100, int((src["score"] + 5) * 20)))
            st.markdown(
                f"**{i}.** 📄 `{src['file_name']}` · "
                f"page {src['page']} · relevance ~{relevance}%"
            )
            highlighted = highlight_text(src["text"], query)
            st.markdown(highlighted, unsafe_allow_html=True)
            st.divider()


def save_uploaded_files(uploaded_files, target_folder: str = "data"):
    """Save uploaded files to a local folder.

    :param uploaded_files: Files from st.file_uploader().
    :param target_folder: Destination folder.
    :return: None
    """
    for file in uploaded_files:
        file_path = os.path.join(target_folder, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())


def get_engine_for_files(target_files):
    """Return a chat engine (basic or hybrid) restricted to the given files.

    Cached; rebuilds only when the filter OR the mode changes.

    :param target_files: List of file names (or None for 'all').
    :return: A chat engine, or None if no index.
    """
    if "index" not in st.session_state:
        return None

    use_hybrid = st.session_state.get("use_hybrid", True)
    signature = (
        use_hybrid,
        tuple(sorted(target_files)) if target_files else None,
    )

    if (
        st.session_state.get("engine_sig") != signature
        or "chat_engine" not in st.session_state
    ):
        if use_hybrid:
            engine = get_hybrid_chat_engine(
                st.session_state.index, file_names=target_files
            )
        else:
            engine = get_chat_engine(
                st.session_state.index, file_names=target_files
            )
        st.session_state.chat_engine = engine
        st.session_state.engine_sig = signature

    return st.session_state.chat_engine


@st.dialog("Filename conflict")
def conflict_dialog(conflicts):
    """Modal asking what to do for same-name-different-content files.

    :param conflicts: List of conflicting file names.
    :return: None
    """
    st.write(
        "These files have the **same name** as documents already indexed, "
        "but their **content is different**. What should I do?"
    )
    decisions = {}
    for conflict in conflicts:
        st.markdown(f"**{conflict}**")
        choice = st.radio(
            label=f"Action for '{conflict}'",
            options=["Replace existing", "Keep both (auto-rename)"],
            key=f"conflict_choice_{conflict}",
            horizontal=True,
        )
        decisions[conflict] = (
            "replace" if choice == "Replace existing" else "keep_both"
        )
        st.divider()

    if st.button("Confirm", use_container_width=True):
        st.session_state.conflict_decisions = decisions
        st.session_state.process_now = True
        st.session_state.show_conflict_dialog = False
        st.rerun()


# ===== MODULE LEVEL =====
st.set_page_config(page_title="Chat with your documents", page_icon="📄")
st.title("Chat with your documents")
st.caption("Local RAG with LlamaIndex + Chroma + Ollama")
os.makedirs("data", exist_ok=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Auto-load the existing index once, on first run of the session
if "index" not in st.session_state and not st.session_state.get(
    "load_attempted"
):
    st.session_state.load_attempted = True  # try only once
    with st.spinner("Loading your documents..."):
        index = load_index()
    if index is not None:
        st.session_state.index = index

# ---------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------
with st.sidebar:
    col1, col2 = st.columns([3, 1])
    col1.header("Documents")
    # Small refresh button, top-right, to reload the index from disk
    if col2.button("🔄", help="Reload documents from disk"):
        st.session_state.pop("index", None)
        st.session_state.pop("load_attempted", None)
        st.session_state.pop("chat_engine", None)
        st.session_state.pop("engine_sig", None)
        st.rerun()

    st.divider()

    # --- Stats panel ---
    if "index" in st.session_state:
        with st.expander("📊 Knowledge Base Stats", expanded=False):
            stats = get_index_stats()

            col1, col2 = st.columns(2)
            col1.metric("📚 Documents", stats["total_documents"])
            col2.metric("📦 Chunks", f"{stats['total_chunks']:,}")

            if st.session_state.get("latencies"):
                lats = st.session_state.latencies
                avg = sum(lats) / len(lats)
                col1b, col2b = st.columns(2)
                col1b.metric("⏱️ Avg response", f"{avg:.1f}s")
                col2b.metric("💬 Questions", len(lats))

            st.divider()
            st.caption(f"🦙 **LLM:** {stats['llm_model']}")
            st.caption(f"🔢 **Embeddings:** {stats['embed_model']}")

            if stats["per_file"]:
                st.divider()
                st.caption("**Chunks per document:**")
                sorted_files = sorted(
                    stats["per_file"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                for fname, count in sorted_files:
                    label = fname if len(fname) < 35 else fname[:32] + "..."
                    st.caption(f"• {label}: **{count:,}**")

    st.divider()

    # --- Retrieval mode ---
    st.subheader("🔧 Retrieval mode")
    use_hybrid = st.toggle("🚀 Hybrid + rerank", value=True)
    st.session_state.use_hybrid = use_hybrid

    st.divider()

    # --- Document selection ---
    st.subheader("🧭 Document selection")
    auto_routing = st.toggle("Automatic routing", value=True)
    st.session_state.auto_routing = auto_routing

    if auto_routing:
        st.caption("🤖 The AI picks relevant documents per question.")
        st.session_state.manual_files = None
    else:
        st.caption("👆 Choose which documents to search:")
        if "index" in st.session_state:
            available = list_indexed_files()
            selected = []
            for fname in available:
                label = fname if len(fname) < 40 else fname[:37] + "..."
                if st.checkbox(label, value=True, key=f"src_{fname}"):
                    selected.append(fname)
            st.session_state.manual_files = selected or None
        else:
            st.info("Load or build an index first.")
            st.session_state.manual_files = None

    st.divider()

    # --- Upload new files ---
    st.subheader("Add / update documents")
    uploaded_files = st.file_uploader(
        label="Choose PDF, Word, text or EPUB files",
        type=["pdf", "docx", "txt", "epub"],
        accept_multiple_files=True,
    )

    if st.button("🔨 Add / update documents"):
        if uploaded_files:
            save_uploaded_files(uploaded_files, "data")
            classification, file_hashes = classify_files("data")
            st.session_state.classification = classification
            st.session_state.file_hashes = file_hashes
            if classification["conflicts"]:
                st.session_state.show_conflict_dialog = True
            else:
                st.session_state.conflict_decisions = {}
                st.session_state.process_now = True
            st.rerun()
        else:
            st.warning("Please upload a PDF or Word file first")

    st.divider()

    # --- Clear the conversation ---
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.latencies = []
        if "chat_engine" in st.session_state:
            st.session_state.chat_engine.reset()

# ---------------------------------------------------------------
# CONFLICT DIALOG + PROCESSING
# ---------------------------------------------------------------
if st.session_state.get("show_conflict_dialog"):
    conflict_dialog(st.session_state.classification["conflicts"])

if st.session_state.get("process_now"):
    st.session_state.process_now = False
    with st.spinner("Processing documents..."):
        report = process_files(
            "data",
            st.session_state.classification,
            st.session_state.file_hashes,
            st.session_state.get("conflict_decisions", {}),
        )
    st.session_state.index = load_index()
    st.session_state.messages = []
    st.session_state.pop("chat_engine", None)
    st.session_state.pop("engine_sig", None)

    if report["added"]:
        st.success(f"✅ Added: {', '.join(report['added'])}")
    if report["replaced"]:
        st.info(f"♻️ Replaced: {', '.join(report['replaced'])}")
    if report["kept_both"]:
        st.info(f"🆕 Kept both: {', '.join(report['kept_both'])}")
    if report["skipped"]:
        st.warning(
            f"⏭️ Skipped (already indexed): {', '.join(report['skipped'])}"
        )

# ---------------------------------------------------------------
# MAIN AREA: chat interface
# ---------------------------------------------------------------
# 1) Re-draw the whole conversation (including sources!) on every rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
    # Redraw sources for assistant messages that have them
    if message.get("sources"):
        render_sources(message["sources"], message.get("query", ""))

# 2) The chat input box
if prompt := st.chat_input("Ask something about your documents..."):

    if "index" not in st.session_state:
        st.error("Build or load an index from the sidebar first")
    else:
        # --- Decide which files to search ---
        if st.session_state.get("auto_routing"):
            all_files = list_indexed_files()
            with st.spinner("🧭 Routing question to relevant documents..."):
                target_files = route_question_to_files(prompt, all_files)
        else:
            target_files = st.session_state.get("manual_files")

        engine = get_engine_for_files(target_files)

        # Show + store the user's message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate + STREAM the assistant's answer
        with st.chat_message("assistant"):
            if target_files:
                st.caption(f"🔍 Searching in: {', '.join(target_files)}")
            else:
                st.caption("🔍 Searching in: all documents")

            start = time.time()
            with st.spinner("Searching the documents..."):
                streaming_response = engine.stream_chat(prompt)
            answer = st.write_stream(streaming_response.response_gen)
            elapsed = time.time() - start

            if "latencies" not in st.session_state:
                st.session_state.latencies = []
            st.session_state.latencies.append(elapsed)
            st.caption(f"⏱️ Answered in {elapsed:.1f}s")

            # Extract sources into simple dicts (so they persist in history)
            sources = []
            for node in streaming_response.source_nodes:
                sources.append({
                    "file_name": node.metadata.get("file_name", "unknown"),
                    "page": node.metadata.get("page_label", "—"),
                    "score": node.score if node.score is not None else 0.0,
                    "text": node.text,
                })

            # Show sources for this fresh answer
            render_sources(sources, query=prompt)

        # Store BOTH content AND sources in history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "query": prompt,
        })

        # Rerun so the sidebar stats reflect this question too
        st.rerun()