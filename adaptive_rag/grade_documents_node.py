from typing import List, Dict, Any
from langchain_core.documents import Document
from My_RAG_Project.utils.log_utils import log
from My_RAG_Project.adaptive_rag.grader_chain import simple_relevance_score, filter_low_score


def grade_documents_node(state):
    """
    Grade and filter retrieved documents.
    - Use the numeric score (metadata['_score']) to compute an average relevance.
    - Filter low-score docs; return need_more_docs if coverage seems insufficient.
    """
    log.info("[Adaptive] grade_documents_node")
    docs: List[Document] = state.get("docs", [])
    avg = simple_relevance_score(docs)
    filtered = filter_low_score(docs, min_score=0.2, min_keep=3)

    need_more_docs = (len(filtered) < 2) or (avg < 0.15)
    quality: Dict[str, Any] = state.get("quality", {})
    quality.update({"relevancy_avg": avg, "kept": len(filtered), "total": len(docs)})

    state["filtered_docs"] = filtered
    state["quality"] = quality
    state["need_more_docs"] = need_more_docs
    return {"filtered_docs": filtered, "quality": quality, "need_more_docs": need_more_docs}
