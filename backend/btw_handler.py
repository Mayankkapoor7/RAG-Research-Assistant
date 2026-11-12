import os
from typing import Generator
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from backend.gateway import get_gateway_llm
from backend.models import BtwRouteDecision

load_dotenv()

llm = get_gateway_llm(model="gpt-4o-mini")


def handle_btw(query: str) -> Generator[str, None, None]:
    """Off-topic side channel — never touches the vector store or checkpointer."""
    route_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Decide if answering this question requires a real-time web search (recent events, "
         "current prices, breaking news) or if your general knowledge is sufficient."),
        ("human", "{query}"),
    ])
    try:
        decision = (route_prompt | llm.with_structured_output(BtwRouteDecision)).invoke({"query": query})
    except Exception:
        decision = BtwRouteDecision(needs_web_search=False)

    if decision.needs_web_search and os.environ.get("TAVILY_API_KEY"):
        try:
            client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
            results = client.search(query, max_results=3)
            context = "\n\n".join(r["content"] for r in results.get("results", []))
            sources = "\n".join(f"- {r['url']}" for r in results.get("results", []))

            answer_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "Answer the question using the web search results below. Be concise.\n\n"
                 f"Results:\n{context}\n\nSources:\n{sources}"),
                ("human", "{query}"),
            ])
        except Exception:
            answer_prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer the question concisely from your general knowledge."),
                ("human", "{query}"),
            ])
    else:
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question concisely from your general knowledge."),
            ("human", "{query}"),
        ])

    for chunk in (answer_prompt | llm).stream({"query": query}):
        if chunk.content:
            yield chunk.content
