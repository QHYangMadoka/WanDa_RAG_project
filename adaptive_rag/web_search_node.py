from typing import List
from langchain_core.documents import Document
from My_RAG_Project.utils.log_utils import log
from My_RAG_Project.llm_models.all_llm import web_search_tool


def web_search_node(state):
    """
    Simple web search fallback via Tavily/Bing/etc.
    Convert results to Document-like objects and append to docs.
    """
    log.info("[Adaptive] web_search_node")
    query = state.get("query") or state.get("user_input") or ""
    if not query:
        return {}

    results = web_search_tool.invoke({"query": query})  # TavilyResults format
    docs: List[Document] = state.get("docs", [])

    for r in results or []:
        content = (r.get("content") or r.get("snippet") or "")[:1200]
        url = r.get("url") or r.get("source") or ""
        docs.append(Document(page_content=content, metadata={"source": url, "page_number": -1, "keywords": "web"}))

    state["docs"] = docs
    return {"docs": docs}
