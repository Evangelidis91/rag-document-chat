import os
import streamlit as st
from rag_engine import (
    classify_files, process_files, load_index,
    get_chat_engine, get_hybrid_chat_engine,   # <- πρόσθεσε αυτό
    list_indexed_files, route_question_to_files,
)


def save_uploaded_files(uploaded_files, target_folder: str = "data"):
    """Save the files uploaded through Streamlit to a local folder so that the
    indexer can read them from disk.
    :param uploaded_files: List of files coming from st.file_uploader().
    :param target_folder: Folder where the files will be written.
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
    # Signature now includes the mode, so switching mode rebuilds too
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
    """Modal asking the user what to do for each file that has the same name as

    an existing document but different content.

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


# ===== MODULE LEVEL: runs every time Streamlit loads the file =====

# Page configuration
st.set_page_config(page_title="Chat with your documents", page_icon="📄")
st.title("Chat with your documents")
st.caption("Local RAG with LlamaIndex + Chroma + Ollama")

# Create the 'data' folder if it does not exist
os.makedirs("data", exist_ok=True)

# Initialise the chat history once
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------
#  SIDEBAR
# ---------------------------------------------------------------
with st.sidebar:
    st.header("Documents")

    # --- Load previously indexed documents (no cost, instant) ---
    st.subheader("Load previous documents")
    st.caption(
        "Reconnect to documents indexed earlier. No embedding, no cost."
    )
    if st.button("📂 Load existing index"):
        with st.spinner("Loading saved index..."):
            index = load_index()
        if index is not None:
            # Store the INDEX (we build engines per-filter on demand)
            st.session_state.index = index
            st.session_state.messages = []
            # Reset cached engine so it rebuilds with current settings
            st.session_state.pop("chat_engine", None)
            st.session_state.pop("engine_sig", None)
            st.success("Loaded! You can start chatting.")
        else:
            st.warning(
                "No saved index found. Upload files and build one first."
            )



    st.divider()
    st.subheader("🔧 Retrieval mode")
    use_hybrid = st.toggle("🚀 Hybrid + rerank", value=True)
    st.session_state.use_hybrid = use_hybrid

    st.divider()

    # --- Search mode (the boolean toggle!) ---
    st.subheader("🧭 Document selection")
    auto_routing = st.toggle("Automatic routing", value=True)
    st.session_state.auto_routing = auto_routing

    # CONDITIONAL: checkboxes only in manual mode
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

    st.divider()

    # --- Upload new files ---
    st.subheader("Add / update documents")
    uploaded_files = st.file_uploader(
        label="Choose PDF or Word files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if st.button("🔨 Add / update documents"):
        if uploaded_files:
            save_uploaded_files(uploaded_files, "data")

            # PHASE 1: classify (cheap, no embeddings)
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
        if "chat_engine" in st.session_state:
            st.session_state.chat_engine.reset()

# ---------------------------------------------------------------
#  CONFLICT DIALOG + PROCESSING
# ---------------------------------------------------------------

# Open the conflict dialog if there are conflicts to resolve
if st.session_state.get("show_conflict_dialog"):
    conflict_dialog(st.session_state.classification["conflicts"])

# PHASE 2: process (this is where embeddings actually happen)
if st.session_state.get("process_now"):
    st.session_state.process_now = False

    with st.spinner("Processing documents..."):
        report = process_files(
            "data",
            st.session_state.classification,
            st.session_state.file_hashes,
            st.session_state.get("conflict_decisions", {}),
        )
        # Reload the index and reset cached engine
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
#  MAIN AREA: chat interface
# ---------------------------------------------------------------

# 1) Re-draw the whole conversation on every rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 2) The chat input box
if prompt := st.chat_input("Ask something about your documents..."):

    if "index" not in st.session_state:
        st.error("Build or load an index from the sidebar first")
    else:
        # --- Decide which files to search ---
        if st.session_state.get("auto_routing"):
            # AUTO: the LLM router picks the relevant files
            all_files = list_indexed_files()
            with st.spinner("🧭 Routing question to relevant documents..."):
                target_files = route_question_to_files(prompt, all_files)
        else:
            # MANUAL: use the user's checkbox selection
            target_files = st.session_state.get("manual_files")

        # --- Get the right engine (cached unless the filter changed) ---
        engine = get_engine_for_files(target_files)

        # Show + store the user's message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate + show the assistant's answer
        with st.chat_message("assistant"):
            # Tell the user which documents are being searched
            if target_files:
                st.caption(f"🔍 Searching in: {', '.join(target_files)}")
            else:
                st.caption("🔍 Searching in: all documents")

            with st.spinner("Searching the documents..."):
                response = engine.chat(prompt)
                answer = str(response)
            st.write(answer)

            # Show the sources used for this answer
            if response.source_nodes:
                with st.expander("Sources"):
                    for i, node in enumerate(response.source_nodes, 1):
                        file_name = node.metadata.get("file_name", "unknown")
                        score = node.score if node.score is not None else 0.0
                        st.markdown(
                            f"**Source {i}** — 📄 {file_name} (score: {score:.2f})"
                        )
                        st.caption(node.text[:300] + "...")

        # Store the assistant's message in the history
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )