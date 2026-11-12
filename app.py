import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from backend.btw_handler import handle_btw
from backend.gateway import get_gateway_llm
from backend.paper_loader import load_arxiv, load_document, load_webpage
from backend.rag_graph import build_graph
from backend.vector_store import add_paper, list_papers

st.set_page_config(
    page_title="Intelligent Multi-Document RAG System",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def get_graph():
    return build_graph()


SESSIONS_FILE = Path("sessions.json")
_rename_llm = get_gateway_llm(model="gpt-4o-mini", temperature=0.0)


def load_sessions() -> dict:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_sessions(sessions_meta: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")


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
                {
                    "content": d.page_content[:300],
                    "metadata": d.metadata,
                }
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
                        "Generate a concise 3-5 word title for a document chat session "
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
                chats.append({
                    "role": "assistant",
                    "content": content,
                    "turn": turn,
                    "graph_state": {},
                    "source": "Session History",
                    "source_documents": [],
                })
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

# ── Bootstrap State ────────────────────────────────────────────────────────────
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
    st.title("🗂️ Knowledge Base")
    if st.button("➕ New Chat Session", use_container_width=True):
        new_sid = create_session()
        st.session_state.active_session_id = new_sid
        active_sid = new_sid
        st.rerun()

    st.markdown("### 💬 Active Sessions")
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
        if uploaded_files:
            processed_key = f"processed_files_{active_sid}"
            if processed_key not in st.session_state:
                st.session_state[processed_key] = set()
            with st.spinner("Processing and chunking files…"):
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
                        docs = load_document(tmp_path, original_filename=f.name)
                        add_paper(docs, active_sid)
                        st.session_state[processed_key].add(f.name)
                        st.success(f"Successfully indexed: {f.name}")
                    except Exception as e:
                        st.error(f"Failed to process {f.name}: {e}")
                    finally:
                        if tmp_path:
                            Path(tmp_path).unlink(missing_ok=True)
            st.rerun()
        else:
            st.warning("Please select at least one file to upload.")

    # ── Section 2: Loaded Documents ───────────────────────────────────────────
    st.divider()
    st.markdown("### 📑 Loaded in this Session")
    try:
        doc_titles = list_papers(active_sid)
    except Exception:
        doc_titles = None

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

st.divider()

# ── Chat display ───────────────────────────────────────────────────────────────
for msg in st.session_state.chats.get(active_sid, []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            source = msg.get("source")
            if source:
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
    is_btw = prompt.strip().lower().startswith("/btw")

    if is_btw:
        query = prompt.strip()[4:].strip()
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption("Side channel — not saved to persistent session history.")

        with st.chat_message("assistant"):
            if not query:
                st.markdown("Please add a question after `/btw`, e.g. `/btw What is transformer attention?`")
            else:
                placeholder = st.empty()
                response_text = ""
                for chunk in handle_btw(query):
                    response_text += chunk
                    placeholder.markdown(response_text + "▌")
                placeholder.markdown(response_text)
            st.caption("Side channel — not saved to persistent session history.")

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
            "contextualized_query": None,
            "route": None,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "claim_verdict": None,
            "claim_source": None,
            "superseding_papers": [],
            "answer": None,
            "is_relevant": None,
            "rewrite_count": 0,
            "source_documents": [],
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

                    placeholder.markdown(response_text + "▌")

            final_state = graph.get_state(config)
            final_values = final_state.values if final_state else {}

            if not response_text:
                response_text = final_values.get("answer") or "No response generated."

            placeholder.markdown(response_text)

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
                st.json(state_snapshot)

        st.session_state.chats[active_sid].append(
            {
                "role": "assistant",
                "content": response_text,
                "graph_state": state_snapshot,
                "turn": current_turn,
                "source": answer_source,
                "source_documents": source_docs,
            }
        )

        if is_first_message:
            st.rerun()
