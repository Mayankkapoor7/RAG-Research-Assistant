import os
import sqlite3
import warnings
import logging
from typing import Annotated

warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.types import Command
from pydantic import BaseModel, Field
from tavily import TavilyClient

from backend.guardrails import REFUSAL_MESSAGE, check_input, check_output
from backend.models import ClaimVerificationResult, RelevancyDecision, RouterDecision
from backend.vector_store import search as vs_search, list_papers
from backend.gateway import get_gateway_llm
from backend import redis_cache
from backend.redis_cache import REDIS_HIT_THRESHOLD, REDIS_MISS_THRESHOLD

load_dotenv()
logger = logging.getLogger(__name__)

llm = get_gateway_llm(model="gpt-4o-mini", temperature=0.0)

# ── State ─────────────────────────────────────────────────────────────────────

class RAGState(MessagesState):
    session_id: str
    query: str
    contextualized_query: str | None
    route: str | None
    retrieved_docs: list[Document]
    retrieval_attempts: int
    claim_verdict: str | None
    claim_source: str | None
    superseding_papers: list[dict] | None
    answer: str | None
    is_relevant: bool | None
    rewrite_count: int
    conversation_summary: str | None
    cache_score: float | None
    answer_source: str | None
    source_documents: list[dict] | None


# ── Contextualize Query Helper ────────────────────────────────────────────────

def contextualize_query(query: str, messages: list) -> str:
    """Reformulate follow-up queries into a standalone query using conversation context."""
    human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    if len(human_msgs) <= 1:
        return query

    recent_dialogue = []
    for m in messages[-6:-1]:
        if isinstance(m, HumanMessage):
            recent_dialogue.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            recent_dialogue.append(f"Assistant: {m.content[:250]}")

    if not recent_dialogue:
        return query

    history_str = "\n".join(recent_dialogue)
    prompt = (
        "Given the conversation history and the latest user question, rephrase the question "
        "to be a standalone search query that contains all necessary entities, names, and topics. "
        "Do NOT answer the question. Return ONLY the standalone query.\n\n"
        f"Conversation History:\n{history_str}\n\n"
        f"Latest Question: {query}\n\n"
        "Standalone Search Query:"
    )
    try:
        res = llm.invoke([{"role": "user", "content": prompt}])
        rewritten = res.content.strip()
        if rewritten and len(rewritten) > 3:
            return rewritten
    except Exception as e:
        logger.debug("Query contextualization skipped: %s", e)
    return query


# ── Router ────────────────────────────────────────────────────────────────────

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a routing assistant for an intelligent Multi-Document RAG system.\n"
        "The system holds enterprise documents, annual/financial reports, operational manuals, policies, and technical reports.\n\n"
        "Classify the user query into exactly one of three categories:\n\n"
        "  retrieve — Use this for ALL questions seeking information, metrics, policies, summaries, "
        "guidance, or comparisons from the uploaded documents (e.g. 'What was the cloud revenue in 2023?', "
        "'What is the password policy?', 'What are the incident SLAs?', 'What are the main risks?').\n"
        "  verify_claim — Use this when the user asks to verify or fact-check whether a claim "
        "has been superseded or updated by recent literature.\n"
        "  direct_answer — ONLY for conversational greetings (e.g. 'hello', 'hi', 'thank you', 'who are you') "
        "or pure generic definitions that require no document context.\n\n"
        "When in doubt, ALWAYS prefer 'retrieve'.\n"
        "Return only the route field.",
    ),
    ("human", "{query}"),
])

router_chain = ROUTER_PROMPT | llm.with_structured_output(RouterDecision)


def router_node(state: RAGState) -> dict:
    query = state["messages"][-1].content
    try:
        decision: RouterDecision = router_chain.invoke({"query": query})
        return {"route": decision.route}
    except Exception as e:
        logger.warning("Router fallback to retrieve: %s", e)
        return {"route": "retrieve"}


# ── Tool schemas ──────────────────────────────────────────────────────────────

class RetrieverInput(BaseModel):
    query: str = Field(description="Semantic query to search document chunks")
    k: int = Field(default=4, ge=1, le=10, description="Number of chunks to retrieve")


class WebSearchInput(BaseModel):
    optimized_query: str = Field(description="Query rewritten and optimized for web search")
    max_results: int = Field(default=3, ge=1, le=10, description="Number of web results to return")


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool(args_schema=RetrieverInput)
def retrieve_from_vectorstore(
    query: str,
    k: int,
    session_id: Annotated[str, InjectedState("session_id")],
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """Search the uploaded document collection (reports, manuals, policies, papers) for relevant passages."""
    docs = vs_search(query=query, session_id=session_id, k=k)
    if not docs:
        return [ToolMessage(content="No relevant documents found in the document collection.", tool_call_id=tool_call_id)]
    summary = f"Retrieved {len(docs)} chunk(s) from uploaded documents."
    return [
        ToolMessage(content=summary, tool_call_id=tool_call_id),
        Command(update={"retrieved_docs": (current_docs or []) + docs}),
    ]


@tool(args_schema=WebSearchInput)
def web_search(
    optimized_query: str,
    max_results: int,
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """Search the web for current or supplementary information using Tavily."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return [ToolMessage(content="Web search unavailable (no API key).", tool_call_id=tool_call_id)]
    try:
        client = TavilyClient(api_key=api_key)
        results = client.search(optimized_query, max_results=max_results)
        if not results.get("results"):
            return [ToolMessage(content="No web results found.", tool_call_id=tool_call_id)]
        web_docs = [
            Document(
                page_content=r["content"],
                metadata={"url": r["url"], "title": r.get("title", "Web Result"), "source": r["url"]},
            )
            for r in results["results"]
        ]
        summary = f"Found {len(web_docs)} web result(s) for: {optimized_query}"
        return [
            ToolMessage(content=summary, tool_call_id=tool_call_id),
            Command(update={"retrieved_docs": (current_docs or []) + web_docs}),
        ]
    except Exception as e:
        return [ToolMessage(content=f"Web search error: {e}", tool_call_id=tool_call_id)]


# ── Retrieval agent singletons ────────────────────────────────────────────────

RETRIEVAL_TOOLS = [retrieve_from_vectorstore, web_search]
retrieval_llm = llm.bind_tools(RETRIEVAL_TOOLS, parallel_tool_calls=False)
base_tool_node = ToolNode(RETRIEVAL_TOOLS)

RETRIEVE_SYSTEM = (
    "You are an intelligent retrieval assistant gathering relevant passages to answer questions about uploaded documents.\n\n"
    "Available tools:\n"
    "1. retrieve_from_vectorstore — searches the uploaded document collection (reports, manuals, policies, papers).\n"
    "   - query: phrase the search query with specific terms, entities, and keywords\n"
    "   - k: number of chunks (default 4)\n\n"
    "2. web_search — searches the live web via Tavily (use ONLY when live external info is needed).\n\n"
    "Rule:\n"
    "- For questions about the uploaded documents, ALWAYS call retrieve_from_vectorstore.\n"
    "- Call only ONE tool per turn.\n"
    "- Do NOT generate a final answer here. Only call tools to retrieve context."
)


# ── Relevancy check ───────────────────────────────────────────────────────────

RELEVANCY_CHECK_SYSTEM = (
    "You are evaluating whether retrieved document chunks are relevant enough "
    "to answer a user's question.\n\n"
    "Return is_relevant=true if the chunks contain information that meaningfully addresses "
    "the question — even partially. "
    "Return is_relevant=false only if the chunks are completely off-topic or empty.\n\n"
    "Be lenient: if there is any substantive overlap or keyword match, return true."
)

relevancy_llm = llm.with_structured_output(RelevancyDecision)

QUERY_REWRITE_SYSTEM = (
    "You are a query rewriting assistant for a document retrieval system. "
    "The previous query failed to retrieve relevant document chunks. "
    "Rewrite the query using more specific synonyms, alternative terms, or keywords.\n\n"
    "Return ONLY the rewritten query as plain text. No explanation."
)


# ── Rolling summary helpers ───────────────────────────────────────────────────

TURNS_TO_KEEP = 8

SUMMARIZE_SYSTEM = (
    "You are a conversation summarizer for an enterprise and research document Q&A assistant. "
    "Given a conversation history, produce a concise paragraph capturing the key topics discussed, "
    "documents referenced, questions asked, and conclusions reached."
)


def _split_messages_for_summary(messages: list, turns_to_keep: int = TURNS_TO_KEEP):
    human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_indices) <= turns_to_keep:
        return [], messages
    cutoff = human_indices[-turns_to_keep]
    return messages[:cutoff], messages[cutoff:]


def agent_node(state: RAGState) -> dict:
    current_attempts = state.get("retrieval_attempts", 0)
    lm = llm if current_attempts >= MAX_RETRIEVAL_ATTEMPTS else retrieval_llm

    query = state.get("query", "")
    messages = state.get("messages", [])
    contextualized = contextualize_query(query, messages)

    system_content = (
        f"{RETRIEVE_SYSTEM}\n\n"
        f"Active Query Context: {contextualized}\n"
        f"Original User Question: {query}"
    )
    summary = state.get("conversation_summary")
    if summary:
        system_content += f"\n\n## Earlier conversation summary:\n{summary}"

    _, recent_msgs = _split_messages_for_summary(messages)
    chat_msgs = [{"role": "system", "content": system_content}] + recent_msgs
    response = lm.invoke(chat_msgs)
    updates: dict = {"messages": [response], "contextualized_query": contextualized}
    if getattr(response, "tool_calls", None):
        updates["retrieval_attempts"] = current_attempts + 1
    return updates


def relevancy_check_node(state: RAGState) -> dict:
    query = state.get("contextualized_query") or state["query"]
    docs = state.get("retrieved_docs") or []
    if not docs:
        return {"is_relevant": False}
    doc_snippets = "\n\n---\n\n".join(doc.page_content[:300] for doc in docs[:3])
    prompt = (
        f"Question: {query}\n\nRetrieved chunks:\n{doc_snippets}\n\n"
        "Are these chunks relevant to answering the question?"
    )
    try:
        decision: RelevancyDecision = relevancy_llm.invoke([
            {"role": "system", "content": RELEVANCY_CHECK_SYSTEM},
            {"role": "user", "content": prompt},
        ])
        return {"is_relevant": decision.is_relevant}
    except Exception:
        return {"is_relevant": True}


def query_rewrite_node(state: RAGState) -> dict:
    original_query = state.get("contextualized_query") or state["query"]
    rewrite_count = state.get("rewrite_count", 0)
    response = llm.invoke([
        {"role": "system", "content": QUERY_REWRITE_SYSTEM},
        {"role": "user", "content": f"Original query: {original_query}\n\nWrite an improved search query."},
    ])
    rewritten = response.content.strip()
    return {
        "messages": [HumanMessage(content=rewritten)],
        "contextualized_query": rewritten,
        "retrieved_docs": [],
        "retrieval_attempts": 0,
        "rewrite_count": rewrite_count + 1,
        "is_relevant": None,
    }


CLAIM_ANALYSIS_PROMPT = (
    "You are a research fact-checker. Given a claim and search results, determine:\n"
    "1. Has this claim been superseded, challenged, or updated by more recent work?\n"
    "2. Identify up to 3 papers from the provided results that supersede or update the claim.\n\n"
    "Rules:\n"
    "- Use ONLY titles and URLs from the provided search results.\n"
    "- If the claim still holds, set is_superseded=false and return an empty superseding_papers list.\n"
    "- verdict_summary should be 1-2 sentences."
)

verification_llm = llm.with_structured_output(ClaimVerificationResult)


def verify_claim_node(state: RAGState) -> dict:
    claim = state["messages"][-1].content
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {
            "claim_verdict": "Web verification unavailable (no Tavily API key configured).",
            "claim_source": None,
            "superseding_papers": [],
        }

    tavily_client = TavilyClient(api_key=api_key)
    general_results = tavily_client.search(f"recent research superseding: {claim[:200]}", max_results=5).get("results", [])
    arxiv_results = tavily_client.search(f"site:arxiv.org {claim[:200]}", max_results=5).get("results", [])

    lines = ["=== General Web Search Results ==="]
    for r in general_results:
        lines.append(f"Title: {r.get('title', '')}\nURL: {r['url']}\nSnippet: {r.get('content', '')[:300]}\n")

    lines.append("=== arXiv Paper Search Results ===")
    for r in arxiv_results:
        lines.append(f"Title: {r.get('title', '')}\nURL: {r['url']}\nSnippet: {r.get('content', '')[:300]}\n")

    context = "\n".join(lines)
    prompt = f"{CLAIM_ANALYSIS_PROMPT}\n\nClaim to verify:\n{claim}\n\nSearch Results:\n{context}"
    result: ClaimVerificationResult = verification_llm.invoke([{"role": "user", "content": prompt}])

    papers_dicts = [p.model_dump() for p in result.superseding_papers[:3]]
    return {
        "claim_verdict": result.verdict_summary,
        "claim_source": papers_dicts[0]["url"] if papers_dicts else None,
        "superseding_papers": papers_dicts,
    }


def _determine_answer_source(route: str | None, docs: list[Document]) -> str:
    if route == "direct_answer":
        return "🧠 General Knowledge"
    if route == "verify_claim":
        return "✅ Fact Check · Web Search"
    if route == "retrieve":
        doc_names = list(dict.fromkeys(
            doc.metadata.get("file_name") or doc.metadata.get("title") or "Document"
            for doc in docs
        ))
        if doc_names:
            names_str = ", ".join(doc_names[:2])
            if len(doc_names) > 2:
                names_str += f" (+{len(doc_names) - 2} more)"
            return f"📄 {names_str}"
        return "📄 Uploaded Documents"
    return "🧠 General Knowledge"


def generate_answer_node(state: RAGState) -> dict:
    route = state.get("route")
    query = state["query"]
    docs = state.get("retrieved_docs") or []
    messages = state.get("messages", [])

    source_records = []

    if route == "retrieve":
        loaded_titles = list_papers(state.get("session_id", ""))
        if not loaded_titles and not docs:
            answer = (
                "The requested information is not available because no documents have been uploaded to this session yet.\n\n"
                "Please upload sample reports (such as annual reports, technical reports, or operational manuals) using the sidebar."
            )
            return {
                "answer": answer,
                "messages": [AIMessage(content=answer)],
                "answer_source": "⚠️ No Documents Uploaded",
                "source_documents": [],
            }

        if state.get("is_relevant") is False and state.get("rewrite_count", 0) >= 1:
            answer = (
                "The requested information is not available in the uploaded documents. "
                "The documents do not contain sufficient information to answer your question. "
                "Please verify that the relevant document is uploaded or rephrase your question."
            )
            return {
                "answer": answer,
                "messages": [AIMessage(content=answer)],
                "answer_source": "📄 Uploaded Documents",
                "source_documents": [],
            }

        if not docs:
            answer = (
                "The requested information is not available in the uploaded documents. "
                "No matching content was found for your query."
            )
            return {
                "answer": answer,
                "messages": [AIMessage(content=answer)],
                "answer_source": "📄 Uploaded Documents",
                "source_documents": [],
            }

        # Format context with explicit document names and page numbers
        formatted_context = []
        seen_sources = set()
        for doc in docs:
            file_name = doc.metadata.get("file_name") or doc.metadata.get("title") or "Document"
            page_num = doc.metadata.get("page")
            page_str = f"Page {page_num}" if page_num else "Page 1"
            header = f"[Document: {file_name} | {page_str}]"
            formatted_context.append(f"{header}\n{doc.page_content}")

            src_key = f"{file_name} - {page_str}"
            if src_key not in seen_sources:
                seen_sources.add(src_key)
                source_records.append({
                    "file_name": file_name,
                    "page": page_num or 1,
                    "snippet": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""),
                })

        context_text = "\n\n---\n\n".join(formatted_context)

        system_instruction = (
            "You are an intelligent assistant in a Multi-Document RAG system.\n"
            "Your task is to answer the user's question accurately, grounded strictly in the provided document context and conversation history.\n\n"
            "MANDATORY REQUIREMENTS:\n"
            "1. Grounding: Answer based ONLY on facts directly stated in the context. Do NOT hallucinate, extrapolate, or assume unstated facts.\n"
            "2. Handling Unavailable Information: If the provided document context does NOT contain enough information to answer the question, explicitly state:\n"
            "   'The requested information is not available in the uploaded documents.'\n"
            "   State clearly what is missing without making up answers.\n"
            "3. Source References: Always include a dedicated '### Sources' section at the end of your answer. "
            "List every document name and page number referenced (e.g. '- TechCorp_Annual_Financial_Report_2023.pdf, Page 2').\n"
            "4. Conversational Context: Maintain continuity with the prior conversation when responding to follow-up questions.\n\n"
            f"## Retrieved Document Context:\n{context_text}"
        )

        # Include prior conversation history for conversational context
        _, recent_history = _split_messages_for_summary(messages)
        history_msgs = []
        for m in recent_history[:-1]:
            if isinstance(m, (HumanMessage, AIMessage)) and m.content:
                history_msgs.append({
                    "role": "user" if isinstance(m, HumanMessage) else "assistant",
                    "content": m.content if isinstance(m.content, str) else str(m.content),
                })

        llm_messages = [{"role": "system", "content": system_instruction}] + history_msgs + [{"role": "user", "content": query}]
        raw_answer = llm.invoke(llm_messages).content
        answer = raw_answer

    elif route == "verify_claim":
        verdict = state.get("claim_verdict", "")
        papers = state.get("superseding_papers") or []
        claim_text = state["query"]
        if papers:
            papers_block = "\n\n".join(
                f"{i + 1}. **{p['title']}**\n   {p['summary']}\n   Link: {p['url']}"
                for i, p in enumerate(papers)
            )
            answer = (
                f"**Claim Verification Result**\n\n"
                f"> {claim_text}\n\n"
                f"**Verdict:** {verdict}\n\n"
                f"**Superseding Papers:**\n\n{papers_block}\n\n"
                f"---\n"
                f"*You can load any of these papers into your knowledge base to continue your research.*"
            )
        else:
            answer = (
                f"**Claim Verification Result**\n\n"
                f"> {claim_text}\n\n"
                f"**Verdict:** {verdict}\n\n"
                f"*No papers directly superseding this claim were found in recent literature.*"
            )

    else:
        prompt = f"Answer concisely and helpfully.\n\nQuestion: {query}"
        answer = llm.invoke([{"role": "user", "content": prompt}]).content

    # Output guard
    is_safe, _ = check_output(answer)
    if not is_safe:
        answer = REFUSAL_MESSAGE

    source = _determine_answer_source(route, docs)
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "answer_source": source,
        "source_documents": source_records,
    }


def maybe_summarize_node(state: RAGState) -> dict:
    messages = state.get("messages", [])
    old_msgs, _ = _split_messages_for_summary(messages)
    if not old_msgs:
        return {}

    existing_summary = state.get("conversation_summary")
    old_text = "\n".join(
        f"{type(m).__name__}: {m.content[:400]}"
        for m in old_msgs
        if hasattr(m, "content") and isinstance(m.content, str) and m.content.strip()
    )

    if existing_summary:
        prompt = (
            f"Existing summary:\n{existing_summary}\n\n"
            f"Older conversation to incorporate:\n{old_text}\n\n"
            "Update the summary to include the older messages above."
        )
    else:
        prompt = f"Summarize this document assistant conversation:\n\n{old_text}"

    try:
        summary_response = llm.invoke([
            {"role": "system", "content": SUMMARIZE_SYSTEM},
            {"role": "user", "content": prompt},
        ])
        return {"conversation_summary": summary_response.content}
    except Exception:
        return {}


MAX_RETRIEVAL_ATTEMPTS = 3


def route_query(state: RAGState) -> str:
    return state.get("route", "retrieve")


def agent_routing(state: RAGState) -> str:
    tc = tools_condition(state)
    if tc == "tools":
        return "retrieval"
    if state.get("retrieval_attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS:
        return "generate_answer"
    return "relevancy_check"


def after_relevancy_routing(state: RAGState) -> str:
    if state.get("is_relevant", False):
        return "generate_answer"
    if state.get("rewrite_count", 0) < 1:
        return "query_rewrite"
    return "generate_answer"


# ── Input guard ──────────────────────────────────────────────────────────────

def input_guard_node(state: RAGState) -> dict:
    question = state["messages"][-1].content
    is_safe, _ = check_input(question)
    if not is_safe:
        return {
            "answer": REFUSAL_MESSAGE,
            "messages": [AIMessage(content=REFUSAL_MESSAGE)],
            "route": "blocked",
        }
    return {}


def after_input_guard(state: RAGState) -> str:
    return "blocked" if state.get("route") == "blocked" else "redis_cache"


# ── Redis semantic cache nodes ─────────────────────────────────────────────

def redis_cache_node(state: RAGState) -> dict:
    query = state["query"]
    try:
        result = redis_cache.get(query)
    except Exception:
        result = None

    if result is None:
        return {"cache_score": 0.0}

    answer, score = result
    if score >= REDIS_HIT_THRESHOLD:
        return {
            "cache_score": score,
            "answer": answer,
            "route": "cache_hit",
            "messages": [AIMessage(content=answer)],
            "answer_source": f"⚡ Semantic Cache · similarity {score:.2f}",
        }

    return {"cache_score": score}


def after_redis_cache(state: RAGState) -> str:
    return "cache_hit" if state.get("route") == "cache_hit" else "router"


def store_to_cache_node(state: RAGState) -> dict:
    route = state.get("route")
    if route in ("cache_hit", "blocked"):
        return {}
    answer = state.get("answer")
    query = state.get("query")
    if answer and query:
        try:
            redis_cache.store(query, answer)
        except Exception:
            pass
    return {}


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(db_path: str = "checkpoints.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = StateGraph(RAGState)

    graph.add_node("input_guard", input_guard_node)
    graph.add_node("redis_cache_node", redis_cache_node)
    graph.add_node("router", router_node)
    graph.add_node("agent_node", agent_node)
    graph.add_node("retrieval", base_tool_node)
    graph.add_node("relevancy_check", relevancy_check_node)
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("verify_claim", verify_claim_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("store_to_cache", store_to_cache_node)
    graph.add_node("maybe_summarize", maybe_summarize_node)

    graph.set_entry_point("input_guard")

    graph.add_conditional_edges(
        "input_guard",
        after_input_guard,
        {"redis_cache": "redis_cache_node", "blocked": END},
    )

    graph.add_conditional_edges(
        "redis_cache_node",
        after_redis_cache,
        {"cache_hit": "maybe_summarize", "router": "router"},
    )

    graph.add_conditional_edges(
        "router",
        route_query,
        {
            "retrieve": "agent_node",
            "verify_claim": "verify_claim",
            "direct_answer": "generate_answer",
        },
    )

    graph.add_conditional_edges(
        "agent_node",
        agent_routing,
        {
            "retrieval": "retrieval",
            "relevancy_check": "relevancy_check",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("retrieval", "agent_node")

    graph.add_conditional_edges(
        "relevancy_check",
        after_relevancy_routing,
        {"query_rewrite": "query_rewrite", "generate_answer": "generate_answer"},
    )
    graph.add_edge("query_rewrite", "agent_node")

    graph.add_edge("verify_claim", "generate_answer")
    graph.add_edge("generate_answer", "store_to_cache")
    graph.add_edge("store_to_cache", "maybe_summarize")
    graph.add_edge("maybe_summarize", END)

    return graph.compile(checkpointer=checkpointer)
