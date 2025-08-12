from typing import List, Tuple
from langchain_core.documents import Document
from My_RAG_Project.utils.log_utils import log
from My_RAG_Project.tools.search_tools import (
    dense_similarity_search,
    sparse_bm25_search,
    hybrid_rrf_search,
)


def retriever_node(state):
    """
    Run retrieval against Milvus PDF collection according to retrieval_params.
    Write distances/scores into metadata['_score'] for grading.
    """
    log.info("[Adaptive] retriever_node")
    query = state.get("query") or state.get("user_input") or ""
    params = state.get("retrieval_params") or {}
    strategy = params.get("strategy", "hybrid")
    k = params.get("k", 5)
    expr = params.get("expr", "page_number >= 1")

    rows: List[Tuple[Document, float]] = []
    if strategy == "dense":
        rows = dense_similarity_search(query, k=k, expr=expr, with_score=True)
    elif strategy == "bm25":
        rows = sparse_bm25_search(query, k=k, expr=expr, with_score=True)
    else:
        rows = hybrid_rrf_search(query, k=k, rrf_k=params.get("rrf_k", 60), expr=expr, with_score=True)

    docs: List[Document] = []
    for doc, score in rows:
        doc.metadata = dict(doc.metadata or {})
        doc.metadata["_score"] = float(score)
        docs.append(doc)

    state["docs"] = docs
    return {"docs": docs}
