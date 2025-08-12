from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from My_RAG_Project.llm_models.embeddings_model import llm
from My_RAG_Project.utils.log_utils import log


def _context_from_docs(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        meta = d.metadata or {}
        page = meta.get("page_number", "")
        src = meta.get("source", "")
        kw = meta.get("keywords", "")
        sc = meta.get("_score", "")
        parts.append(f"[p{page} | score={sc} | kw={kw} | src={src}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def _citations(docs: List[Document]) -> List[Dict[str, Any]]:
    cites = []
    for d in docs:
        meta = d.metadata or {}
        cites.append({
            "page_number": meta.get("page_number"),
            "source": meta.get("source"),
            "keywords": meta.get("keywords"),
            "score": meta.get("_score"),
            "snippet": (d.page_content[:180] + "...") if d.page_content else ""
        })
    return cites


def generate_node2(state):
    """
    Generate final answer strictly from filtered_docs.
    """
    log.info("[Adaptive] generate_node2")
    docs: List[Document] = state.get("filtered_docs", [])
    question: str = state.get("query") or state.get("user_input") or ""

    if not docs:
        state["answer"] = "抱歉，未检索到足够的上下文来回答该问题。"
        state["citations"] = []
        return {"answer": state["answer"], "citations": []}

    context = _context_from_docs(docs)
    prompt = PromptTemplate(
        template=(
            "You are a helpful Chinese assistant. Answer the question ONLY using the given context.\n"
            "If the context is insufficient, say you don't know.\n\n"
            "Question:\n{question}\n\n"
            "Context:\n{context}\n\n"
            "Answer in Chinese:"
        ),
        input_variables=["question", "context"],
    )
    chain = prompt | llm | StrOutputParser()
    ans = chain.invoke({"question": question, "context": context})

    state["answer"] = ans
    state["citations"] = _citations(docs)
    return {"answer": ans, "citations": state["citations"]}
