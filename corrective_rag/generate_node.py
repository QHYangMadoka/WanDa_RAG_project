from typing import List, Any

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from My_RAG_Project.corrective_rag.get_human_message import get_last_human_message
from My_RAG_Project.llm_models.embeddings_model import llm
from My_RAG_Project.utils.log_utils import log


def _stringify_tool_output(docs: Any) -> str:
    """
    Normalize tool output into a readable string context.

    - If it's a list of LangChain Documents, join their page_content with separators,
      and preserve selected metadata (page_number, keywords).
    - If it's already a string, return it directly.
    - Else, str() fallback.
    """
    try:
        from langchain_core.documents import Document  # optional import
    except Exception:
        Document = None

    if isinstance(docs, list) and docs and (Document is None or isinstance(docs[0], object)):
        parts: List[str] = []
        for d in docs:
            # Robust extraction to avoid attribute errors
            content = getattr(d, "page_content", None) or str(d)
            meta = getattr(d, "metadata", {}) or {}
            page = meta.get("page_number", "")
            kw = meta.get("keywords", "")
            parts.append(f"[p{page} | kw={kw}]\n{content}")
        return "\n\n---\n\n".join(parts)
    if isinstance(docs, str):
        return docs
    return str(docs)


def generate(state):
    """
    Generate the final answer based on retrieved context and the user's question.
    """
    log.info("--- generating final answer ---")
    messages = state["messages"]
    question = get_last_human_message(messages).content
    last_message = messages[-1]

    raw_context = getattr(last_message, "content", "")
    context = _stringify_tool_output(raw_context)

    # Prompt template
    prompt = PromptTemplate(
        template=(
            "You are a helpful QA assistant. Answer the question using ONLY the context.\n"
            "If the answer is not present, say you don't know.\n\n"
            "Question:\n{question}\n\n"
            "Context:\n{context}\n\n"
            "Answer:"
        ),
        input_variables=["question", "context"],
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": question})
    return {"messages": [AIMessage(content=response)]}
