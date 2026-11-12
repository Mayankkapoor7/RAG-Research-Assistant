import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage
<<<<<<< HEAD

from backend.btw_handler import handle_btw
from backend.gateway import get_gateway_llm
=======
from langchain_openai import ChatOpenAI

from backend.btw_handler import handle_btw
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
from backend.paper_loader import load_arxiv, load_document, load_webpage
from backend.rag_graph import build_graph
from backend.vector_store import add_paper, list_papers

<<<<<<< HEAD
st.set_page_config(
    page_title="Intelligent Multi-Document RAG System",
    page_icon="📚",
    layout="wide",
)
=======
st.set_page_config(page_title="Papeer", page_icon="📚", layout="centered")
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6


@st.cache_resource
def get_graph():
    return build_graph()


SESSIONS_FILE = Path("sessions.json")
<<<<<<< HEAD
_rename_llm = get_gateway_llm(model="gpt-4o-mini", temperature=0.0)


=======
_rename_llm = ChatOpenAI(model="gpt-3.5-turbo")

# loads the sessions
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
def load_sessions() -> dict:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

<<<<<<< HEAD

def save_sessions(sessions_meta: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")


=======
# Save the session in a particular format
def save_sessions(sessions_meta: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")

# We are serializing the python object into the json 
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
def _serialize_state(values: dict) -> dict:
    out = {}
    for k, v in values.items():
        if k == "messages":
            out[k] = [
                {
                    "type": type(m).__name__,
                    "content": (
                        m.content[:300]
                        if isinstance(m.content, str)
                        else repr(m.content)[:300]
                    ),
                }
                for m in (v or [])
            ]
        elif k == "retrieved_docs":
            out[k] = [
<<<<<<< HEAD
                {
                    "content": d.page_content[:300],
                    "metadata": d.metadata,
                }
=======
                {"content": d.page_content[:300], "metadata": d.metadata}
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
                for d in (v or [])
            ]
        else:
            out[k] = v
    return out


def generate_session_name(first_message: str) -> str:
    try:
        response = _rename_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
<<<<<<< HEAD
                        "Generate a concise 3-5 word title for a document chat session "
=======
                        "Generate a concise 3-5 word title for a research chat session "
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
                        "based on the user's first message. Return only the title, "
                        "no punctuation at the end, no quotes."
                    ),
                },
                {"role": "user", "content": first_message[:500]},
            ]
        )
        return response.content.strip()
    except Exception:
        return "New Session"


def maybe_rename_session(session_id: str, first_message: str) -> None:
    if st.session_state.sessions_meta.get(session_id, {}).get("is_named"):
        return
    name = generate_session_name(first_message)
    st.session_state.sessions_meta[session_id]["name"] = name
    st.session_state.sessions_meta[session_id]["is_named"] = True
    save_sessions(st.session_state.sessions_meta)


def create_session() -> str:
    sid = str(uuid.uuid4())
    st.session_state.sessions_meta[sid] = {
        "id": sid,
        "name": "New Session",
        "created_at": datetime.now().isoformat(),
        "is_named": False,
    }
    save_sessions(st.session_state.sessions_meta)
    st.session_state.chats[sid] = []
    st.session_state.turns[sid] = 0
    return sid


def load_session_chats(session_id: str) -> list[dict]:
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        if not state or not state.values:
            return []
        chats = []
        turn = 0
        for msg in state.values.get("messages", []):
            type_name = type(msg).__name__
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if type_name == "HumanMessage":
                chats.append({"role": "user", "content": content})
            elif type_name in ("AIMessage", "AIMessageChunk"):
                turn += 1
<<<<<<< HEAD
                chats.append({
                    "role": "assistant",
                    "content": content,
                    "turn": turn,
                    "graph_state": {},
                    "source": "Session History",
                    "source_documents": [],
                })
=======
                chats.append({"role": "assistant", "content": content, "turn": turn, "graph_state": {}})
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
        return chats
    except Exception:
        return []


def switch_session(session_id: str) -> None:
    st.session_state.active_session_id = session_id
    if session_id not in st.session_state.chats:
        st.session_state.chats[session_id] = load_session_chats(session_id)
    if session_id not in st.session_state.turns:
        turn_count = sum(1 for m in st.session_state.chats[session_id] if m["role"] == "assistant")
        st.session_state.turns[session_id] = turn_count


graph = get_graph()

<<<<<<< HEAD
# ── Bootstrap State ────────────────────────────────────────────────────────────
=======
# ── Bootstrap ──────────────────────────────────────────────────────────────────
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
if "sessions_meta" not in st.session_state:
    st.session_state.sessions_meta = load_sessions()
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "turns" not in st.session_state:
    st.session_state.turns = {}
if "active_session_id" not in st.session_state:
    if st.session_state.sessions_meta:
        latest = max(
            st.session_state.sessions_meta.values(),
            key=lambda s: s["created_at"],
        )
        switch_session(latest["id"])
    else:
        sid = create_session()
        st.session_state.active_session_id = sid

active_sid = st.session_state.active_session_id

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
<<<<<<< HEAD
    st.title("🗂️ Knowledge Base")
    if st.button("➕ New Chat Session", use_container_width=True):
=======
    if st.button("+ New Chat", use_container_width=True):
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
        new_sid = create_session()
        st.session_state.active_session_id = new_sid
        active_sid = new_sid
        st.rerun()
<<<<<<< HEAD

    st.markdown("### 💬 Active Sessions")
=======
    st.divider()
    st.markdown("## 💬 Sessions")

>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
    sorted_sessions = sorted(
        st.session_state.sessions_meta.values(),
        key=lambda s: s["created_at"],
        reverse=True,
    )
    for session in sorted_sessions:
        sid = session["id"]
        is_active = sid == st.session_state.active_session_id
        btn_type = "primary" if is_active else "secondary"
        if st.button(
            session["name"],
            key=f"sess_{sid}",
            use_container_width=True,
            type=btn_type,
        ):
            if not is_active:
                switch_session(sid)
                st.rerun()

    st.divider()
<<<<<<< HEAD

    # ── Sample Reports Quick-Load ──────────────────────────────────────────────
    st.markdown("### 📊 Sample Reports")
    st.caption("Load sample reports (Financial, Operational, Technical) for instant testing.")
    if st.button("⚡ Load All 3 Sample Reports", use_container_width=True, type="secondary"):
        with st.spinner("Indexing sample reports..."):
            sample_docs = [
                Path("documents/TechCorp_Annual_Financial_Report_2023.pdf"),
                Path("documents/Enterprise_Cloud_Security_Operational_Manual.pdf"),
                Path("documents/Openclaw_Research_Report.pdf"),
            ]
            loaded_count = 0
            for doc_path in sample_docs:
                if doc_path.exists():
                    try:
                        docs = load_document(str(doc_path), original_filename=doc_path.name)
                        add_paper(docs, active_sid)
                        loaded_count += 1
                    except Exception as e:
                        st.error(f"Error loading {doc_path.name}: {e}")
            if loaded_count > 0:
                st.success(f"Indexed {loaded_count} sample reports into session!")
                st.rerun()

    st.divider()

    # ── Section 1: File Upload ─────────────────────────────────────────────────
    st.markdown("### 📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or Markdown documents",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        key=f"uploader_{active_sid}",
        help="Upload multiple PDF reports, policies, manuals, or technical papers.",
    )
    if st.button("📥 Index Uploaded Files", use_container_width=True, key="btn_add_files"):
=======
    st.markdown("## 📄 Documents")

    # ── Section 1: File upload ─────────────────────────────────────────────────
    st.markdown("**Upload Files**")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or Markdown",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        key=f"uploader_{active_sid}",
        label_visibility="collapsed",
    )
    if st.button("Add Files", use_container_width=True, key="btn_add_files"):
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
        if uploaded_files:
            processed_key = f"processed_files_{active_sid}"
            if processed_key not in st.session_state:
                st.session_state[processed_key] = set()
<<<<<<< HEAD
            with st.spinner("Processing and chunking files…"):
=======
            with st.spinner("Processing files…"):
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
                for f in uploaded_files:
                    if f.name in st.session_state[processed_key]:
                        st.info(f"Already loaded: {f.name}")
                        continue
                    suffix = Path(f.name).suffix
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(f.read())
                            tmp_path = tmp.name
<<<<<<< HEAD
                        docs = load_document(tmp_path, original_filename=f.name)
                        add_paper(docs, active_sid)
                        st.session_state[processed_key].add(f.name)
                        st.success(f"Successfully indexed: {f.name}")
                    except Exception as e:
                        st.error(f"Failed to process {f.name}: {e}")
=======
                        docs = load_document(tmp_path)
                        for doc in docs:
                            doc.metadata["title"] = Path(f.name).stem
                        add_paper(docs, active_sid)
                        st.session_state[processed_key].add(f.name)
                        st.success(f"Added: {f.name}")
                    except Exception as e:
                        st.error(f"Failed: {f.name} — {e}")
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
                    finally:
                        if tmp_path:
                            Path(tmp_path).unlink(missing_ok=True)
            st.rerun()
        else:
<<<<<<< HEAD
            st.warning("Please select at least one file to upload.")

    # ── Section 2: Loaded Documents ───────────────────────────────────────────
    st.divider()
    st.markdown("### 📑 Loaded in this Session")
=======
            st.warning("No files selected.")

    # ── Section 2: Web URL loader ──────────────────────────────────────────────
    st.markdown("**Web Pages**")
    url_input = st.text_area(
        "URLs (one per line)",
        key=f"url_area_{active_sid}",
        height=80,
        label_visibility="collapsed",
        placeholder="https://example.com/paper",
    )
    if st.button("Load URLs", use_container_width=True, key="btn_load_urls"):
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        if urls:
            with st.spinner("Loading web pages…"):
                for url in urls:
                    try:
                        docs = load_webpage(url)
                        add_paper(docs, active_sid)
                        st.success(f"Loaded: {url[:60]}")
                    except Exception as e:
                        st.error(f"Failed: {url[:60]} — {e}")
            st.rerun()
        else:
            st.warning("Enter at least one URL.")

    # ── Section 3: ArXiv loader ────────────────────────────────────────────────
    st.markdown("**ArXiv Papers**")
    arxiv_title = st.text_input(
        "Paper title or ArXiv ID",
        key=f"arxiv_input_{active_sid}",
        label_visibility="collapsed",
        placeholder="1706.03762  or  Attention Is All You Need",
    )
    if st.button("Load ArXiv Paper", use_container_width=True, key="btn_load_arxiv"):
        if arxiv_title.strip():
            with st.spinner("Loading from ArXiv…"):
                try:
                    docs = load_arxiv(arxiv_title.strip())
                    add_paper(docs, active_sid)
                    loaded_title = docs[0].metadata.get("title") if docs else arxiv_title.strip()
                    st.success(f"Loaded: {loaded_title}")
                except Exception as e:
                    st.error(f"Failed: {e}")
            st.rerun()
        else:
            st.warning("Enter a paper title or ArXiv ID.")

    # ── Loaded Documents list ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### Loaded Documents")
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
    try:
        doc_titles = list_papers(active_sid)
    except Exception:
        doc_titles = None
<<<<<<< HEAD

    if doc_titles:
        for title in doc_titles:
            st.markdown(f"- 📄 **{title}**")
    elif doc_titles is None:
        st.caption("Unable to fetch document index.")
    else:
        st.info("No documents loaded yet in this session. Click 'Load All 3 Sample Reports' or upload files above.")


# ── Main Chat Interface ────────────────────────────────────────────────────────
st.title("📚 Intelligent Multi-Document RAG System")
st.markdown(
    "Ask natural language questions across your uploaded technical reports, financial reports, "
    "operational manuals, and business policies. Answers are strictly **grounded**, include **source references & page citations**, "
    "maintain **conversational memory** for follow-ups, and gracefully **handle unavailable information**."
)

# ── Sample Questions Helper ────────────────────────────────────────────────────
with st.expander("💡 Click to view sample test questions across multiple documents", expanded=False):
    st.markdown("""
- **Financial Report Query:** `What was TechCorp's total consolidated revenue and cloud operating margin in FY2023?`
- **Follow-up / Conversational Memory:** `What about its growth rate and R&D investments compared to other segments?`
- **Operational Manual Query:** `What are the password requirements and the response SLA for Severity 1 incidents?`
- **Technical Report Query:** `What are the primary benchmark results and architecture features of OpenClaw?`
- **Unavailable Information Test:** `What is the warranty and return policy for retail consumer smartphones?`
- **Fact-Check / Verification:** `verify claim: Transformers have completely replaced convolutional neural networks in all computer vision tasks.`
""")

=======
    if doc_titles is None:
        st.caption("Could not load document list — try refreshing.")
    elif doc_titles:
        for title in doc_titles:
            st.markdown(f"- {title}")
    else:
        st.caption("No documents loaded yet.")

# ── Page header ────────────────────────────────────────────────────────────────
st.title("📚 Papeer — Research Paper Assistant")
st.markdown(
    "🔍 **Ask questions** from your uploaded papers &nbsp;·&nbsp; "
    "✅ **Verify claims** against recent literature &nbsp;·&nbsp; "
    "🌐 **Search the web** for the latest findings\n\n"
    "> Upload documents in the sidebar and start chatting below."
)
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
st.divider()

# ── Chat display ───────────────────────────────────────────────────────────────
for msg in st.session_state.chats.get(active_sid, []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            source = msg.get("source")
            if source:
<<<<<<< HEAD
                st.caption(f"**Source:** {source}")

            source_docs = msg.get("source_documents") or []
            if source_docs:
                with st.expander("📖 Grounded Source Excerpts & Citations", expanded=False):
                    for idx, s in enumerate(source_docs, 1):
                        st.markdown(f"**[{idx}] {s.get('file_name', 'Document')} (Page {s.get('page', 1)})**")
                        st.markdown(f"> *\"{s.get('snippet', '')}\"*")

            graph_state = msg.get("graph_state")
            if graph_state:
                with st.expander(f"📊 Execution Trace · Turn {msg.get('turn', 1)}", expanded=False):
                    st.json(graph_state)

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about the documents or ask a follow-up…"):
=======
                st.caption(f"Source: {source}")
            with st.expander(f"📊 Graph state · turn {msg['turn']}", expanded=False):
                st.json(msg["graph_state"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about your papers, verify a claim, or search the web…"):
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
    is_btw = prompt.strip().lower().startswith("/btw")

    if is_btw:
        query = prompt.strip()[4:].strip()
<<<<<<< HEAD
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption("Side channel — not saved to persistent session history.")

        with st.chat_message("assistant"):
            if not query:
                st.markdown("Please add a question after `/btw`, e.g. `/btw What is transformer attention?`")
=======

        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption("Side channel — not saved to session history.")

        with st.chat_message("assistant"):
            if not query:
                st.markdown("Please add a question after `/btw`, e.g. `/btw What is attention?`")
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
            else:
                placeholder = st.empty()
                response_text = ""
                for chunk in handle_btw(query):
                    response_text += chunk
                    placeholder.markdown(response_text + "▌")
                placeholder.markdown(response_text)
<<<<<<< HEAD
            st.caption("Side channel — not saved to persistent session history.")
=======
            st.caption("Side channel — not saved to session history.")
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6

    else:
        if active_sid not in st.session_state.chats:
            st.session_state.chats[active_sid] = []
        if active_sid not in st.session_state.turns:
            st.session_state.turns[active_sid] = 0

        is_first_message = len(st.session_state.chats[active_sid]) == 0

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chats[active_sid].append({"role": "user", "content": prompt})
        st.session_state.turns[active_sid] += 1
        current_turn = st.session_state.turns[active_sid]

        if is_first_message:
            maybe_rename_session(active_sid, prompt)

        input_state = {
            "messages": [HumanMessage(content=prompt)],
            "session_id": active_sid,
            "query": prompt,
<<<<<<< HEAD
            "contextualized_query": None,
=======
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
            "route": None,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "claim_verdict": None,
            "claim_source": None,
            "superseding_papers": [],
            "answer": None,
            "is_relevant": None,
            "rewrite_count": 0,
<<<<<<< HEAD
            "source_documents": [],
=======
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
        }
        config = {"configurable": {"thread_id": active_sid}}

        with st.chat_message("assistant"):
            placeholder = st.empty()
            response_text = ""

            for chunk, metadata in graph.stream(input_state, config, stream_mode="messages"):
                if (
                    metadata.get("langgraph_node") == "generate_answer"
                    and hasattr(chunk, "content")
                    and chunk.content
                ):
                    if type(chunk).__name__ == "AIMessageChunk":
                        response_text += chunk.content
                    elif type(chunk).__name__ == "AIMessage":
                        response_text = chunk.content
<<<<<<< HEAD

                    placeholder.markdown(response_text + "▌")

            final_state = graph.get_state(config)
            final_values = final_state.values if final_state else {}

            if not response_text:
=======
                    
                    placeholder.markdown(response_text + "▌")

            if not response_text:
                final_values = graph.get_state(config).values
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
                response_text = final_values.get("answer") or "No response generated."

            placeholder.markdown(response_text)

<<<<<<< HEAD
            answer_source = final_values.get("answer_source")
            if answer_source:
                st.caption(f"**Source:** {answer_source}")

            source_docs = final_values.get("source_documents") or []
            if source_docs:
                with st.expander("📖 Grounded Source Excerpts & Citations", expanded=False):
                    for idx, s in enumerate(source_docs, 1):
                        st.markdown(f"**[{idx}] {s.get('file_name', 'Document')} (Page {s.get('page', 1)})**")
                        st.markdown(f"> *\"{s.get('snippet', '')}\"*")

            state_snapshot = _serialize_state(final_values)
            with st.expander(f"📊 Execution Trace · Turn {current_turn}", expanded=False):
=======
            final_values = graph.get_state(config).values
            answer_source = final_values.get("answer_source")
            if answer_source:
                st.caption(f"Source: {answer_source}")

            state_snapshot = _serialize_state(final_values)

            with st.expander(f"📊 Graph state · turn {current_turn}", expanded=False):
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
                st.json(state_snapshot)

        st.session_state.chats[active_sid].append(
            {
                "role": "assistant",
                "content": response_text,
                "graph_state": state_snapshot,
                "turn": current_turn,
                "source": answer_source,
<<<<<<< HEAD
                "source_documents": source_docs,
=======
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
            }
        )

        if is_first_message:
            st.rerun()
