from typing import List
from langchain_core.documents import Document


def simple_relevance_score(docs: List[Document]) -> float:
    """
    Compute an average score from metadata['_score'] if present.
    """
    if not docs:
        return 0.0
    scores = [float(d.metadata.get("_score", 0.0)) for d in docs]
    return sum(scores) / max(len(scores), 1)


def filter_low_score(docs: List[Document], min_score: float = 0.2, min_keep: int = 3) -> List[Document]:
    """
    Keep top documents above a threshold, fallback to top-N if all low.
    """
    if not docs:
        return []
    docs_sorted = sorted(docs, key=lambda d: float(d.metadata.get("_score", 0.0)), reverse=True)
    kept = [d for d in docs_sorted if float(d.metadata.get("_score", 0.0)) >= min_score]
    if kept:
        return kept[:max(min_keep, len(kept))]
    return docs_sorted[:min_keep]
