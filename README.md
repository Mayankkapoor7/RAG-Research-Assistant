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
```

---

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
