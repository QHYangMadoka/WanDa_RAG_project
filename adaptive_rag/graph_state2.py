from typing import List, Dict, Any, TypedDict, Annotated, Optional
from langchain_core.documents import Document
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


class AdaptiveState(TypedDict, total=False):
    """
    Global state object carried across Adaptive RAG graph.

    messages: dialog/tool messages accumulated by the graph.
    user_input: the original user query.
    query: query used for retrieval after transform.
    retrieval_params: parameters for retrieval (strategy/k/expr/rrf_k).
    docs: raw retrieved documents.
    filtered_docs: documents after grading/filtering.
    answer: final answer text.
    citations: list of citation dictionaries (source/page/snippet/score).
    needs_web: whether to route to web search.
    quality: dict to store quality signals (relevance/coverage/hallucination scores).
    iterations: loop counter to avoid infinite retries.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    user_input: str
    query: str
    retrieval_params: Dict[str, Any]
    docs: List[Document]
    filtered_docs: List[Document]
    answer: str
    citations: List[Dict[str, Any]]
    needs_web: bool
    quality: Dict[str, Any]
    iterations: int


def default_retrieval_params() -> Dict[str, Any]:
    """
    Default retrieval parameters for PDF+Milvus stack.
    """
    return {
        "strategy": "hybrid",          # "dense" | "bm25" | "hybrid"
        "k": 5,
        "rrf_k": 60,
        "expr": "page_number >= 1"
    }


def inc_iterations(state: AdaptiveState, max_iters: int = 3) -> bool:
    """
    Increase iteration counter and return True if the loop should stop.
    """
    iters = state.get("iterations", 0) + 1
    state["iterations"] = iters
    return iters >= max_iters
