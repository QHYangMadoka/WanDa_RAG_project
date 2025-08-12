from My_RAG_Project.utils.log_utils import log
from My_RAG_Project.llm_models.embeddings_model import llm
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class AnswerQuality(BaseModel):
    relevance: float = Field(description="0~1; is the answer relevant to the question?")
    uses_citations: bool = Field(description="Does the answer reflect the provided context?")
    sufficient: bool = Field(description="Is the answer sufficient to address the user question?")


def grade_answer_chain(state):
    """
    Grade final answer quality using LLM. You can replace with rules if needed.
    If quality is low, signal a retry.
    """
    log.info("[Adaptive] grade_answer_chain")
    answer = state.get("answer", "")
    docs = state.get("filtered_docs", [])
    question = state.get("query") or state.get("user_input") or ""

    context_preview = "\n\n---\n\n".join([d.page_content[:150] for d in docs])

    prompt = PromptTemplate.from_template(
        "Question:\n{q}\n\nContext Preview:\n{c}\n\nAnswer:\n{a}\n\n"
        "Evaluate:\n"
        "- relevance (0~1)\n"
        "- uses_citations (true/false)\n"
        "- sufficient (true/false)\n"
        "Return JSON with keys: relevance, uses_citations, sufficient"
    )
    structured = llm.with_structured_output(AnswerQuality)
    quality = structured.invoke(prompt.format(q=question, c=context_preview, a=answer))

    state.setdefault("quality", {})
    state["quality"]["answer_relevance"] = quality.relevance
    state["quality"]["answer_uses_citations"] = quality.uses_citations
    state["quality"]["answer_sufficient"] = quality.sufficient

    retry = (quality.relevance < 0.5) or (not quality.sufficient)
    state["need_retry_answer"] = retry
    return {"quality": state["quality"], "need_retry_answer": retry}
