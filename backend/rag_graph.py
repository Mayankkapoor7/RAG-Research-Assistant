import os
import sqlite3
import warnings
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
from backend.vector_store import search as vs_search
from backend.gateway import get_gateway_llm
from backend import redis_cache
from backend.redis_cache import REDIS_HIT_THRESHOLD, REDIS_MISS_THRESHOLD

load_dotenv()

llm = get_gateway_llm()

# ── State ─────────────────────────────────────────────────────────────────────

class RAGState(MessagesState):
    session_id: str
    query: str
    route: str | None
    retrieved_docs: list[Document]
    retrieval_attempts: int
    claim_verdict: str | None
    claim_source: str | None
    superseding_papers: list[dict] | None
    answer: str | None
    is_relevant: bool | None # context is relevant to query or not
    rewrite_count: int # how many times we reqrote the query
    conversation_summary: str | None  # rolling summary of turns older than TURNS_TO_KEEP
    cache_score: float | None  # cosine similarity score from Redis semantic cache lookup
    answer_source: str | None    # human-readable label: where the answer came from


# ── Router ────────────────────────────────────────────────────────────────────

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a routing assistant for a research paper Q&A system. "
        "Classify the user query into exactly one of three categories:\n\n"
        "  retrieve — Use this for TWO types of questions:\n"
        "    (a) Questions about the content of uploaded research papers "
        "(e.g. methods, results, conclusions, authors).\n"
        "    (b) Questions that require live or current information that cannot be "
        "answered from general knowledge alone — such as current events, today's weather, "
        "live prices, recent news, or anything where the answer changes over time "
        "(e.g. 'Who is the current president?', 'What is the price of gold today?', "
        "'What is the weather in Delhi?').\n"
        "  verify_claim — The user wants to check whether a specific claim or finding "
        "from a paper is still accurate or has been superseded.\n"
        "  direct_answer — A stable general knowledge question answerable from training data "
        "with no retrieval needed (e.g. 'What is softmax?', 'Who invented the transformer?', "
        "'Explain backpropagation.').\n\n"
        "When in doubt between retrieve and direct_answer, prefer retrieve.\n\n"
        "Return only the route field.",
    ),
    ("human", "{query}"),
])

router_chain = ROUTER_PROMPT | llm.with_structured_output(RouterDecision)


def router_node(state: RAGState) -> dict:
    query = state["messages"][-1].content # Latest message
    decision: RouterDecision = router_chain.invoke({"query": query})
    return {"route": decision.route} # Returns state with updated route


# ── Tool schemas ──────────────────────────────────────────────────────────────

class RetrieverInput(BaseModel):
    query: str = Field(description="Semantic query to search research paper chunks") # Required
    k: int = Field(default=4, ge=1, le=10, description="Number of chunks to retrieve") # Optional with default


class WebSearchInput(BaseModel):
    optimized_query: str = Field(description="Query rewritten and optimized for web search") # Required
    max_results: int = Field(default=3, ge=1, le=10, description="Number of web results to return") # Optional with default


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool(args_schema=RetrieverInput)
def retrieve_from_vectorstore(
    query: str, # Semantic query to search research paper chunks
    k: int, # Number of chunks to retrieve
    session_id: Annotated[str, InjectedState("session_id")], # Session ID to identify the user
    current_docs: Annotated[list, InjectedState("retrieved_docs")], # Current documents in the state
    tool_call_id: Annotated[str, InjectedToolCallId], # Tool call ID for the tool
) -> list:
    """Search the uploaded research paper vector store for relevant passages."""
    docs = vs_search(query=query, session_id=session_id, k=k)
    if not docs:
        return [ToolMessage(content="No relevant documents found in the vector store.", tool_call_id=tool_call_id)]
    summary = f"Retrieved {len(docs)} chunk(s) from the vector store."
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
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    results = client.search(optimized_query, max_results=max_results)
    if not results.get("results"):
        return [ToolMessage(content="No web results found.", tool_call_id=tool_call_id)]
    web_docs = [
        Document(
            page_content=r["content"],
            metadata={"url": r["url"], "title": r.get("title", "Web Result")},
        )
        for r in results["results"]
    ]
    summary = f"Found {len(web_docs)} web result(s) for: {optimized_query}"
    return [
        ToolMessage(content=summary, tool_call_id=tool_call_id),
        Command(update={"retrieved_docs": (current_docs or []) + web_docs}),
    ]


# ── Retrieval agent singletons ────────────────────────────────────────────────

RETRIEVAL_TOOLS = [retrieve_from_vectorstore, web_search]
retrieval_llm = llm.bind_tools(RETRIEVAL_TOOLS, parallel_tool_calls=False) # You are only allowed to call ONE tool per turn
base_tool_node = ToolNode(RETRIEVAL_TOOLS) # This will execute the tool calls, This is the retrieval node


RETRIEVE_SYSTEM = (
    "You are a research assistant gathering context to answer a user's question about research papers.\n\n"
    "You have two tools available and full control over how you use them:\n\n"
    "1. retrieve_from_vectorstore — searches the uploaded paper collection.\n"
    "   You decide:\n"
    "   - query: the semantic search query (phrase it to best match relevant paper chunks)\n"
    "   - k: how many chunks to retrieve (1–10; use more for broad questions, fewer for specific ones)\n\n"
    "2. web_search — searches the live web via Tavily.\n"
    "   You decide:\n"
    "   - optimized_query: rewrite the user's question as a concise, keyword-rich web search query\n"
    "   - max_results: how many results to fetch (1–10)\n\n"
    "Choose the right source based on the question:\n"
    "- Questions about the uploaded papers → use retrieve_from_vectorstore\n"
    "- Questions about current events, recent developments, or supplementary information → use web_search\n"
    "- Call only one tool per turn.\n\n"
    "Do NOT produce a final answer. Only call tools to collect context."
)


# ── Relevancy check ───────────────────────────────────────────────────────────

RELEVANCY_CHECK_SYSTEM = (
    "You are evaluating whether retrieved document chunks are relevant enough "
    "to answer a user's question about research papers.\n\n"
    "Return is_relevant=true if the chunks contain information that meaningfully "
    "addresses the question — even partially. "
    "Return is_relevant=false only if the chunks are clearly off-topic or contain "
    "no useful information.\n\nBe lenient: if there is any substantive overlap, return true."
)

relevancy_llm = llm.with_structured_output(RelevancyDecision)

QUERY_REWRITE_SYSTEM = (
    "You are a query rewriting assistant for a research paper retrieval system. "
    "The previous query failed to retrieve relevant document chunks. "
    "Rewrite the query using more specific or alternative terminology, "
    "domain-specific keywords, or a narrower sub-question.\n\n"
    "Return ONLY the rewritten query as plain text. No explanation, no preamble."
)


# ── Rolling summary helpers ───────────────────────────────────────────────────

TURNS_TO_KEEP = 10  # number of recent human turns to keep verbatim

SUMMARIZE_SYSTEM = (
    "You are a conversation summarizer for a research paper Q&A assistant. "
    "Given a conversation history, produce a concise paragraph capturing: "
    "the topics discussed, key questions asked, papers or claims referenced, "
    "and important conclusions reached. Be factual and brief."
)


def _split_messages_for_summary(messages: list, turns_to_keep: int = TURNS_TO_KEEP):
    """Split messages into (old_messages, recent_messages) keyed by HumanMessage turns.

    Returns:
        old_msgs   – everything before the last `turns_to_keep` human turns
        recent_msgs – the last `turns_to_keep` turns (all message types included)
    """
    human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_indices) <= turns_to_keep:
        return [], messages  # not enough turns to summarise yet
    cutoff = human_indices[-turns_to_keep]  # index of oldest message to keep verbatim
    return messages[:cutoff], messages[cutoff:]



def agent_node(state: RAGState) -> dict:
    current_attempts = state.get("retrieval_attempts", 0)
    # Once at the cap, use plain LLM so the agent cannot emit more tool calls.
    # This prevents orphaned tool_call IDs from entering the persisted message history.
    # retrieval llm --> tool call --> tool result
    # llm --> no tools are bounded --> tool call
    lm = llm if current_attempts >= MAX_RETRIEVAL_ATTEMPTS else retrieval_llm

    # Use only the recent turn window; prepend rolling summary when available
    _, recent_msgs = _split_messages_for_summary(state["messages"])
    system_content = RETRIEVE_SYSTEM
    summary = state.get("conversation_summary")
    if summary:
        system_content = (
            f"{RETRIEVE_SYSTEM}\n\n"
            f"## Earlier conversation summary:\n{summary}"
        )

    messages = [{"role": "system", "content": system_content}] + recent_msgs
    response = lm.invoke(messages)
    updates: dict = {"messages": [response]}
    if getattr(response, "tool_calls", None):
        updates["retrieval_attempts"] = current_attempts + 1
    return updates



def relevancy_check_node(state: RAGState) -> dict:
    query = state["query"]
    docs = state.get("retrieved_docs") or []
    doc_snippets = "\n\n---\n\n".join(doc.page_content[:300] for doc in docs[:3])
    if not doc_snippets:
        return {"is_relevant": False}
    prompt = (
        f"Question: {query}\n\nRetrieved chunks:\n{doc_snippets}\n\n"
        "Are these chunks relevant to answering the question?"
    )
    decision: RelevancyDecision = relevancy_llm.invoke([
        {"role": "system", "content": RELEVANCY_CHECK_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    return {"is_relevant": decision.is_relevant}


def query_rewrite_node(state: RAGState) -> dict:
    original_query = state["query"]
    rewrite_count = state.get("rewrite_count", 0)
    response = llm.invoke([
        {"role": "system", "content": QUERY_REWRITE_SYSTEM},
        {"role": "user", "content": f"Original query: {original_query}\n\nWrite an improved search query."},
    ])
    rewritten = response.content.strip()
    return {
        "messages": [HumanMessage(content=rewritten)],
        "query": rewritten,
        "retrieved_docs": [], # We want the agent node to restart, discard the irrelevant documents
        "retrieval_attempts": 0,
        "rewrite_count": rewrite_count + 1,
        "is_relevant": None,
    }


CLAIM_ANALYSIS_PROMPT = (
    "You are a research fact-checker. Given a claim from a research paper and "
    "a set of recent web and arXiv search results, determine:\n"
    "1. Has this claim been superseded, significantly challenged, or updated by more recent work?\n"
    "2. Identify up to 3 papers from the provided results that supersede or update the claim.\n\n"
    "Rules:\n"
    "- Use ONLY titles and URLs that appear verbatim in the provided search results.\n"
    "- Prefer arXiv paper links (arxiv.org) over general web links when available.\n"
    "- For each superseding paper, write one sentence explaining how it supersedes the claim.\n"
    "- If the claim still holds, set is_superseded=false and return an empty superseding_papers list.\n"
    "- verdict_summary should be 1-2 sentences suitable for display to the user."
)

verification_llm = llm.with_structured_output(ClaimVerificationResult)


def verify_claim_node(state: RAGState) -> dict:
    claim = state["messages"][-1].content
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    # General web search for recent work superseding the claim
    general_results = tavily_client.search(
        f"recent research superseding: {claim[:200]}",
        max_results=5,
    ).get("results", [])


    arxiv_results = tavily_client.search(
        f"site:arxiv.org {claim[:200]}",
        max_results=5,
    ).get("results", [])

    lines = ["=== General Web Search Results ==="]
    for r in general_results:
        lines.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r.get('content', '')[:300]}\n"
        )

    lines.append("=== arXiv Paper Search Results ===")
    for r in arxiv_results:
        lines.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r.get('content', '')[:300]}\n"
        )

    context = "\n".join(lines)

    prompt = (
        f"{CLAIM_ANALYSIS_PROMPT}\n\n"
        f"Claim to verify:\n{claim}\n\n"
        f"Search Results:\n{context}"
    )
    result: ClaimVerificationResult = verification_llm.invoke([
        {"role": "user", "content": prompt}
    ])

    papers_dicts = [p.model_dump() for p in result.superseding_papers[:3]]
    return {
        "claim_verdict": result.verdict_summary,
        "claim_source": papers_dicts[0]["url"] if papers_dicts else None,
        "superseding_papers": papers_dicts,
    }


def _determine_answer_source(route: str | None, docs: list[Document]) -> str:
    """Map the current route + retrieved docs to a human-readable source label."""
    if route == "direct_answer":
        return "🧠 LLM Knowledge"
    if route == "verify_claim":
        return "✅ Claim Verification  ·  Web Search"
    if route == "retrieve":
        has_web = any(doc.metadata.get("url") for doc in docs)
        has_paper = any(not doc.metadata.get("url") for doc in docs)
        if has_web and has_paper:
            return "📄 Research Papers  ·  🌐 Web Search"
        if has_web:
            return "🌐 Web Search"
        if has_paper:
            return "📄 Research Papers"
        return "📄 Research Papers"  # fallback when docs list is empty
    return "🧠 LLM Knowledge"


def generate_answer_node(state: RAGState) -> dict:
    route = state.get("route")
    query = state["query"]
    docs = state.get("retrieved_docs") or []

    if route == "retrieve":
        if state.get("is_relevant") is False and state.get("rewrite_count", 0) >= 1:
            answer = (
                "I wasn't able to find relevant information in the uploaded papers "
                "to answer your question. You may want to rephrase your question "
                "or upload additional papers."
            )
        else:
            if not docs:
                answer = "I don't know the answer."
            else:
                context = "\n\n---\n\n".join(doc.page_content for doc in docs)
                prompt = f"Answer the question using this context:\n\n{context}\n\nQuestion: {query}"
                answer = llm.invoke([{"role": "user", "content": prompt}]).content

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
                f"*You can load any of these papers into your knowledge base "
                f"to continue your research with the latest findings.*"
            )
        else:
            answer = (
                f"**Claim Verification Result**\n\n"
                f"> {claim_text}\n\n"
                f"**Verdict:** {verdict}\n\n"
                f"*No papers directly superseding this claim were found in recent literature.*"
            )

    else:
        prompt = f"Answer from your knowledge.\n\nQuestion: {query}"
        answer = llm.invoke([{"role": "user", "content": prompt}]).content

    # Output guard — replace answer if it violates the output policy
    is_safe, _ = check_output(answer)
    if not is_safe:
        answer = REFUSAL_MESSAGE

    source = _determine_answer_source(route, docs)
    return {"answer": answer, "messages": [AIMessage(content=answer)], "answer_source": source}


def maybe_summarize_node(state: RAGState) -> dict:
    """After each answer, compress old turns into a rolling summary stored in state.

    The summary lives in RAGState.conversation_summary and is automatically
    persisted to checkpoints.db by SqliteSaver — no separate storage needed.
    Only runs once the conversation exceeds TURNS_TO_KEEP human turns.
    """
    messages = state.get("messages", [])
    old_msgs, _ = _split_messages_for_summary(messages)
    if not old_msgs:
        return {}  # fewer than TURNS_TO_KEEP turns — nothing to summarise yet

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
        prompt = f"Summarize this research assistant conversation:\n\n{old_text}"

    summary_response = llm.invoke([
        {"role": "system", "content": SUMMARIZE_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    return {"conversation_summary": summary_response.content}


MAX_RETRIEVAL_ATTEMPTS = 3


def route_query(state: RAGState) -> str:
    return state["route"]


def agent_routing(state: RAGState) -> str:
    # Always execute pending tool calls first — shortcutting here would leave
    # an AIMessage with tool_calls unmatched by ToolMessages in the checkpointer,
    # corrupting history for all future turns in the same session.
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
    """Block unsafe questions before they reach the router."""
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
    """Check Redis for a semantically similar cached answer.

    Outcomes
    --------
    HIT  (score >= REDIS_HIT_THRESHOLD)  : set answer + messages, mark route='cache_hit'.
    MISS (score <  REDIS_MISS_THRESHOLD) : pass through with cache_score recorded.
    AMBIGUOUS (between thresholds)       : also pass through; freshness wins.
    """
    query = state["query"]
    result = redis_cache.get(query)

    if result is None:
        # Cache empty or Redis unavailable — treat as miss
        return {"cache_score": 0.0}

    answer, score = result

    if score >= REDIS_HIT_THRESHOLD:
        # ── Cache HIT: short-circuit the entire RAG pipeline ──────────────
        return {
            "cache_score": score,
            "answer": answer,
            "route": "cache_hit",
            "messages": [AIMessage(content=answer)],
            "answer_source": f"⚡ Redis Cache  ·  similarity {score:.2f}",
        }

    # MISS or ambiguous zone — fall through to router
    return {"cache_score": score}


def after_redis_cache(state: RAGState) -> str:
    """Route to maybe_summarize on cache hit, router otherwise."""
    return "cache_hit" if state.get("route") == "cache_hit" else "router"


def store_to_cache_node(state: RAGState) -> dict:
    """Persist the freshly generated answer to Redis after the RAG pipeline.

    Skips storage when:
    - route == 'cache_hit'  (answer came from cache; no need to re-store)
    - route == 'blocked'    (guardrail refusal; never cache)
    - answer or query is missing
    """
    route = state.get("route")
    if route in ("cache_hit", "blocked"):
        return {}
    answer = state.get("answer")
    query = state.get("query")
    if answer and query:
        redis_cache.store(query, answer)
    return {}


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(db_path: str = "checkpoints.db"):
    """Compile the full RAG LangGraph with Redis semantic caching.

    Execution order
    ───────────────
    input_guard
        ↓ (safe)
    redis_cache_node ──(HIT score ≥ REDIS_HIT_THRESHOLD)──→ maybe_summarize
        ↓ (MISS / ambiguous)
    router
        ↓
    [retrieve / verify_claim / direct_answer branches]
        ↓
    generate_answer
        ↓
    store_to_cache          ← persists answer; skips on cache_hit / blocked
        ↓
    maybe_summarize → END
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = StateGraph(RAGState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("redis_cache_node", redis_cache_node)   # NEW
    graph.add_node("router", router_node)
    graph.add_node("agent_node", agent_node)
    graph.add_node("retrieval", base_tool_node)
    graph.add_node("relevancy_check", relevancy_check_node)
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("verify_claim", verify_claim_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("store_to_cache", store_to_cache_node)  # NEW
    graph.add_node("maybe_summarize", maybe_summarize_node)

    graph.set_entry_point("input_guard")

    # ── input_guard → redis_cache_node (or END if blocked) ───────────────────
    graph.add_conditional_edges(
        "input_guard",
        after_input_guard,
        {"redis_cache": "redis_cache_node", "blocked": END},
    )

    # ── redis_cache_node → maybe_summarize (HIT) or router (MISS) ────────────
    graph.add_conditional_edges(
        "redis_cache_node",
        after_redis_cache,
        {"cache_hit": "maybe_summarize", "router": "router"},
    )

    # ── router → retrieve / verify_claim / direct_answer ─────────────────────
    graph.add_conditional_edges(
        "router",
        route_query,
        {
            "retrieve": "agent_node",
            "verify_claim": "verify_claim",
            "direct_answer": "generate_answer",
        },
    )

    # ── Retrieval sub-graph ───────────────────────────────────────────────────
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

    # ── Answer → cache storage → summary → END ────────────────────────────────
    graph.add_edge("verify_claim", "generate_answer")
    graph.add_edge("generate_answer", "store_to_cache")    # was: maybe_summarize
    graph.add_edge("store_to_cache", "maybe_summarize")    # NEW
    graph.add_edge("maybe_summarize", END)

    return graph.compile(checkpointer=checkpointer)

