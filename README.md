<<<<<<< HEAD
# Intelligent Multi-Document RAG System (Assignment 3)

An intelligent Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **LangChain**, and **Streamlit** that enables users to upload multiple PDF documents (technical reports, annual/financial reports, operational manuals, policies) and ask natural language questions with context-aware, fully grounded answers and verifiable source references.

---

## 🎯 Assignment 3 Problem Statement & Requirements Fulfillment

| Assignment 3 Requirement | How This System Fulfills It | Status |
|---|---|:---:|
| **Multi-Document PDF Upload** | Allows uploading multiple PDF documents (financial reports, operational manuals, technical reports). Chunks documents and preserves document file names, page numbers (1-indexed), and titles. | ✅ Verified |
| **Accurate Document Retrieval** | Uses semantic vector search powered by `text-embedding-3-small` and persistent Qdrant (auto-fallback to local persistent `./qdrant_db` or Qdrant Cloud) with cosine distance matching. | ✅ Verified |
| **Grounded Response Generation** | Enforces strict LLM grounding rules: answers are synthesized solely from retrieved document context. No ungrounded speculation or hallucination. | ✅ Verified |
| **Relevant Source References** | Every response cites exact source documents and page numbers in a dedicated `### Sources` section and structured UI citations expanders (`[Document: <name>, Page: <page>]`). | ✅ Verified |
| **Handling Unavailable Information** | When the query is outside the scope of uploaded documents, the system explicitly informs the user that the information is unavailable rather than fabricating answers. | ✅ Verified |
| **Conversational Context & Memory** | Maintains full conversation history across multiple turns. Rephrases follow-up queries (resolving pronouns and implicit references) and provides conversation memory to the answer generator. | ✅ Verified |
| **Sample Reports Dataset** | Includes 3 diverse sample reports in `documents/`: Financial (`TechCorp_Annual_Financial_Report_2023.pdf`), Operational/Policy (`Enterprise_Cloud_Security_Operational_Manual.pdf`), and Technical (`Openclaw_Research_Report.pdf`). | ✅ Verified |

---

## 🏗️ System Architecture

```
app.py (Streamlit Interactive Web UI)
│
├── backend/rag_graph.py       — LangGraph RAG Agent Workflow
│     ├── input_guard          — Guardrails input validation
│     ├── redis_cache_node     — Semantic caching layer
│     ├── router               — Intelligent intent classification (retrieve / verify_claim / direct_answer)
│     ├── agent_node           — Contextual query reformulation & retrieval tool execution
│     ├── relevancy_check      — Relevancy verification & automatic query rewrite
│     ├── generate_answer      — Grounded answer generation with source citations & history
│     └── maybe_summarize      — Rolling memory compression for long conversations
│
├── backend/paper_loader.py    — PDF loader with PyMuPDF (stamps original filename & 1-indexed page)
├── backend/vector_store.py    — Resilient Qdrant vector store with Local & Cloud support + embedding cache
├── backend/gateway.py         — LLM gateway with Portkey support and direct OpenAI fallback
├── backend/guardrails.py      — Safety checks against prompt injection and malicious queries
└── backend/btw_handler.py     — Fast side-channel for off-topic queries (`/btw`)
=======
# Agentic RAG Research Paper Assistant

A conversational AI assistant for students and researchers to upload, explore, and verify academic papers through natural language chat.

---

## Project Description

This is a Retrieval-Augmented Generation (RAG) application built with LangGraph, LangChain, and Streamlit. Users upload research papers (PDF, TXT, Markdown, web URL, or ArXiv ID) into isolated sessions, then ask questions about them. The system routes each query intelligently — answering directly from paper content, searching the web for current developments, or verifying whether a claim from a paper has been superseded by newer research.

---

## Target Users

- **Students** reading and trying to understand dense academic papers
- **Researchers** who want to quickly cross-reference claims across multiple papers
- **Literature reviewers** checking whether findings or methods from older papers still hold today
- **Anyone** who wants a conversational interface to a set of documents without manual reading

---

## Features

| Feature | Description |
|---|---|
| **Paper Q&A** | Ask questions about uploaded papers; the system retrieves relevant chunks and generates grounded answers |
| **Claim Verification** | Ask the assistant to verify a claim — it searches the web and ArXiv to determine if the claim is current or superseded, and returns links to newer papers if applicable |
| **Web Search** | For questions about current developments or explicit search requests, live Tavily results are incorporated |
| **Direct Answers** | General knowledge questions are answered without retrieval or web calls |
| **`/btw` Command** | A side-channel for off-topic questions outside the session context. The LLM decides to answer directly or search the web. These exchanges are **not stored in session history** |
| **Multi-session UI** | Open multiple independent sessions simultaneously, each with its own paper collection and conversation history |
| **Auto Session Naming** | Session titles are automatically generated (3–5 words) from the first message using the LLM |
| **Multiple Paper Sources** | Load papers via file upload (PDF, TXT, MD), web URL, or ArXiv ID/title search |
| **Graph State Inspector** | Each assistant turn exposes an expandable JSON view of the LangGraph state for debugging |
| **Streaming Responses** | Assistant responses stream token-by-token with a cursor animation |

---

## How to Use

### 1. Start a session
Launch the app and a default session is created automatically. Use **New Chat** in the sidebar to start additional sessions.

### 2. Upload papers
In the sidebar, choose one of three loading methods:
- **File Upload** — drag and drop a PDF, TXT, or MD file
- **Web URL** — paste one or more URLs (one per line)
- **ArXiv** — enter a paper title or ArXiv ID (e.g. `2303.08774`)

Loaded papers are listed under "Loaded Papers" in the sidebar.

### 3. Ask questions
Type in the chat input. Example queries:
- *"What methodology does the paper use for evaluation?"*
- *"Verify the claim that encoder-decoder models are the best approach for translation."*
- *"What are the latest developments in diffusion models?"*

### 4. Use `/btw` for off-topic questions
Prefix any message with `/btw` to ask a question outside the current paper context. These exchanges are not saved to the session:
```
/btw What is the difference between RLHF and DPO?
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
```

---

<<<<<<< HEAD
## 📄 Included Sample Reports Dataset

The system includes 3 realistic sample PDF reports in the `documents/` directory:

1. **`TechCorp_Annual_Financial_Report_2023.pdf`** (4 pages):
   - Total consolidated revenues ($128.4B, 18.2% YoY growth), operating income ($42.6B), net profit ($34.8B).
   - Division margins (Cloud Platform 38.5%, Enterprise Software 44.2%, Hardware 19.4%).
   - Geographic distribution (North America $68B, EMEA $35B, APAC $25.4B).
   - R&D investments ($22.7B total: $14.2B Generative AI, $5B Quantum Silicon), FY2024 guidance, risk factors.

2. **`Enterprise_Cloud_Security_Operational_Manual.pdf`** (4 pages):
   - Data classification tiers (Tier 1 Public through Tier 4 Secret).
   - Password policy (minimum 14 characters, rotation every 90 days, 12 password history, 5 failed attempts lockout).
   - MFA standards (FIDO2 hardware tokens mandatory, SMS prohibited), RBAC and JIT elevation.
   - Incident severity matrix (Severity 1 SLA: 15-minute response, 72-hour regulatory notification).
   - Disaster recovery SLAs (RTO < 2 hours, RPO < 15 minutes, air-gapped backups) and compliance audit schedules.

3. **`Openclaw_Research_Report.pdf`** (Technical Research Report):
   - Architecture, agentic workflows, autonomous execution benchmarks, and system latency.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) package manager

### 2. Install Dependencies
```bash
# In the project directory:
uv sync
```

### 3. Configure `.env`
Ensure your `.env` contains:
```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
LANGSMITH_TRACING=false
# Optional: QDRANT_URL and QDRANT_API_KEY (if omitted or unreachable, local persistent storage is used automatically)
```

### 4. Run the Streamlit Application
```bash
uv run streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Automated Testing & Verification Suite

A comprehensive test suite is provided in `test_rag_system.py` verifying all 5 core assignment criteria:

```bash
uv run python test_rag_system.py
```

### Verification Suite Tests:
1. **Multi-Document PDF Loading & Metadata Stamping**: Verifies chunking, original filename stamping, and 1-indexed page numbering across multiple PDFs.
2. **Cross-Document Retrieval**: Verifies targeted retrieval routing to the correct PDF.
3. **Grounded Answer Generation & Source Citations**: Confirms answers cite document name and page number.
4. **Conversational Memory & Follow-up**: Confirms follow-up questions resolve pronouns (`"What about its operating margin for the Cloud division?"` -> 38.5%).
5. **Handling Unavailable Information**: Confirms questions outside the documents return clear unavailability messages without hallucination.

---

## 💡 Key Features in the UI

- **One-Click Sample Reports**: Click **⚡ Load All 3 Sample Reports** in the sidebar to immediately index all sample reports.
- **Custom Document Upload**: Upload your own PDF, TXT, or Markdown documents via the file uploader.
- **Source Reference Inspector**: Expand the **📖 Grounded Source Excerpts & Citations** panel under any assistant response to view chunk citations and text quotes.
- **Graph State & Execution Trace**: Expand **📊 Execution Trace** to inspect the internal LangGraph state, retrieved chunk contents, and node progression.
- **Session Management**: Manage independent conversation sessions with auto-generated session titles.
=======
## Installation

The RAG Assistant uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone <repo-url>
cd rag--project

# Install all dependencies
uv sync

# Copy the example env file and fill in your keys
cp .env.example .env

# Run the Streamlit app
uv run streamlit run app.py
```

To add a new dependency:
```bash
uv add <package-name>
```

To run a backend module directly (useful during development):
```bash
uv run python -m backend.<module_name>
```

---

## Required API Keys

All keys are loaded from a `.env` file in the project root via `python-dotenv`.

| Variable | Purpose | Where to Get It |
|---|---|---|
| `OPENAI_API_KEY` | LLM inference (`gpt-5-mini`) and embeddings (`text-embedding-3-small`) | [platform.openai.com](https://platform.openai.com) |
| `TAVILY_API_KEY` | Web search for current developments and claim verification | [tavily.com](https://tavily.com) |
| `QDRANT_URL` | Qdrant Cloud endpoint for the vector store | [cloud.qdrant.io](https://cloud.qdrant.io) |
| `QDRANT_API_KEY` | Authentication for Qdrant Cloud | [cloud.qdrant.io](https://cloud.qdrant.io) |

`.env` file format:
```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
```

---

## Architecture

```
app.py (Streamlit UI)
│
├── backend/rag_graph.py       — LangGraph RAG workflow (router → retrieve/verify/direct → answer)
├── backend/btw_handler.py     — Off-topic /btw handler (streaming, not stored in history)
├── backend/vector_store.py    — Qdrant Cloud vector store with cached embeddings
├── backend/paper_loader.py    — Multi-source paper loader (PDF, TXT, MD, URL, ArXiv)
└── backend/models.py          — Pydantic models for routing and structured LLM outputs
```

### RAG Graph Decision Flow

```
User Query
    │
    ▼
 Router (LLM)
    │
    ├── direct_answer ──────────────────────────► Generate Answer
    │
    ├── retrieve ──► Agent (retriever + web tools) ──► Relevancy Check
    │                        │                              │
    │                        │◄── Query Rewrite (max 3) ────┘
    │                        └──────────────────────────────► Generate Answer
    │
    └── verify_claim ──► Web Search + ArXiv Search ──► Verdict + Paper Links
```

---

## How the Project Is Production Optimized

| Optimization | Details |
|---|---|
| **Embedding cache** | `CacheBackedEmbeddings` writes to `./embedding_cache/` so identical text is never re-embedded across sessions — reduces OpenAI API calls and latency |
| **Session isolation** | Each session gets its own Qdrant collection (`assistant_{session_id}`) and a separate LangGraph SQLite checkpointer thread — prevents cross-session data leakage |
| **Graph caching** | The LangGraph graph is built once with `@st.cache_resource` and reused across all Streamlit reruns |
| **Streaming responses** | `graph.stream()` is used with message mode so responses appear token-by-token rather than waiting for the full generation |
| **Session persistence** | `sessions.json` persists session metadata; SQLite stores full conversation state — app restarts restore the previous session seamlessly |
| **Temp file cleanup** | Uploaded files are written to a temp path, processed, then deleted regardless of success or failure |
| **Async evaluation** | The evaluation pipeline uses throttled concurrency (3 workers, 5 s throttle) to stay within API rate limits |
| **ArXiv reliability** | Claim verification uses two targeted Tavily searches (general web + `site:arxiv.org`) instead of the `arxiv` Python library, which had reliability issues |

---

## Constraints and Why

| Constraint | Why |
|---|---|
| **Max 3 query rewrites** | The RAG graph caps query rewrites at 3 retries before falling back to a plain LLM answer. Without this cap, ambiguous or unanswerable queries would loop indefinitely, burning API tokens and blocking the user |
| **Chunk size 1000 / overlap 200** | Balances retrieval precision (smaller = more focused) against context preservation across chunk boundaries. The 200-char overlap ensures sentences split across chunks are still retrievable |
| **Tavily max 3 results for `/btw`** | Keeps the context window manageable for side-channel queries that are intentionally lightweight and unsaved |
| **`/btw` exchanges not stored** | These are deliberately out-of-context questions. Storing them would pollute session history and confuse the LLM's understanding of the paper-focused conversation |
| **Session-scoped Qdrant collections** | Prevents papers from one session leaking into another. Each collection is namespaced by session UUID |
| **Claim verification uses two searches** | A general web search catches blog posts and news; an `arxiv.org`-targeted search catches academic superseding work. One search alone misses one of these two important source types |
| **`k=4` default retrieval chunks** | Balances context richness against prompt length. Too few chunks miss relevant content; too many dilute focus and increase cost |

---

## Evaluation

The Research Assistant includes an automated RAG evaluation pipeline (`evaluate.py`) built on [DeepEval](https://github.com/confident-ai/deepeval).

### Metrics (threshold: 0.7)

| Metric | What It Measures |
|---|---|
| **Contextual Precision** | Are the retrieved chunks relevant to the query? |
| **Contextual Recall** | Does the retrieved context cover all expected information? |
| **Contextual Relevancy** | Is the context relevant to both the input and the expected output? |
| **Answer Relevancy** | Does the generated answer actually address the question? |
| **Faithfulness** | Is the answer grounded in the retrieved context (no hallucination)? |

### Running Evaluation

```bash
uv run python evaluate.py
```

- On first run, synthetic golden test cases are generated from `documents/Openclaw_Research_Report.pdf` and cached to `goldens.json`
- Results are written to `eval_results.json` with per-test metric scores, pass/fail status, and failure reasons
- Subsequent runs reuse cached goldens unless `goldens.json` is deleted
>>>>>>> 3444167b334963c9af6020b6c471033a498110a6
