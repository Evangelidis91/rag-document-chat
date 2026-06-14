import os
import streamlit as st
from rag_engine import get_chat_engine, process_files, load_index, classify_files, get_hybrid_chat_engine


def build_engine(index):
    """
    Create the right chat engine based on the user's settings:
    advanced (hybrid + rerank) or basic(vector only)
    :param index: The loaded VectorStoreIndex
    :return: A chat engine, or None if no index is loaded
    """
    if st.session_state.get("use_hybrid", True):
        return get_hybrid_chat_engine(
            index=index,
            top_k_retrieve=st.session_state.get("top_k", 10),
            top_n_rerank=st.session_state.get("top_n", 3),
        )
    return get_chat_engine(index=index)


def save_uploaded_files(uploaded_files, target_folder: str = "data"):
    """Save the files uploaded through Streamlit to a local folder so that the
    indexer can read them from disk.
    :param uploaded_files: List of files coming from st.file_uploader().
    :param target_folder: Folder where the files will be written.
    :return: None
    """
    for file in uploaded_files:
        file_path = os.path.join(target_folder, file.name)
        # Write the file in binary mode (works for PDF/DOCX)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

@st.dialog("Filename conflict")
def conflict_dialog(conflicts):
    """
    Modal asking the user what to do for each file that has the same name as
    an existing document but different content.
    :param conflicts: List of conflict file names.
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


# ===== MODULE LEVEL: everything below runs when Streamlit loads the file =====

# Page configuration
st.set_page_config(page_title="Chat with your documents", page_icon="📄")
st.title("Chat with your documents")
st.caption("RAG with LlamaIndex + Chroma")

# Create the 'data' folder if it does not exist
os.makedirs("data", exist_ok=True)

# Initialise the chat history once (a list of {role, content} dicts)
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
            st.session_state.chat_engine = build_engine(index)
            st.session_state.messages = []
            st.success("Loaded! You can start chatting.")
        else:
            st.warning(
                "No saved index found. Upload files and build one first."
            )

    st.divider()
    st.subheader("Retrieval mode")
    use_hybrid = st.toggle("Advanced (hybrid + rerank)", value=True)
    top_k = st.slider("Candidates to retrieve", 5, 20, 10)
    top_n = st.slider("Chunks after rerank", 1, 8, 3)
    st.session_state.use_hybrid = use_hybrid
    st.session_state.top_k = top_k
    st.session_state.top_n = top_n

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
                # Ask the user first via the modal
                st.session_state.show_conflict_dialog = True
            else:
                # Nothing to ask -> go straight to processing
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
#  CONFLICT DIALOG + PROCESSING (between sidebar and chat)
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
        index = load_index()
        st.session_state.chat_engine = build_engine(index)
        st.session_state.messages = []

    # Tell the user exactly what happened
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

# 2) The chat input box (sticks to the bottom, like ChatGPT)
if prompt := st.chat_input("Ask something about your documents..."):

    # Guard: make sure an index has been built/loaded first
    if "chat_engine" not in st.session_state:
        st.error("Build or load an index from the sidebar first")
    else:
        # Show + store the user's message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate + show the assistant's answer
        with st.chat_message("assistant"):
            with st.spinner("Searching the documents..."):
                response = st.session_state.chat_engine.chat(prompt)
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