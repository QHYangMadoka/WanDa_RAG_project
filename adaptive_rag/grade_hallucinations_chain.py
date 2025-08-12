from My_RAG_Project.utils.log_utils import log
from My_RAG_Project.llm_models.embeddings_model import llm
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class HalluCheck(BaseModel):
    hallucination: bool = Field(description="true if the answer includes claims not supported by context")


def grade_hallucinations_chain(state):
    """
    Check whether the answer includes hallucinated claims.
    """
    log.info("[Adaptive] grade_hallucinations_chain")
    answer = state.get("answer", "")
    docs = state.get("filtered_docs", [])
    context = "\n\n---\n\n".join([d.page_content for d in docs])

    prompt = PromptTemplate.from_template(
        "Given the context and the answer, determine if the answer contains unsupported claims.\n\n"
        "Context:\n{c}\n\nAnswer:\n{a}\n\n"
        "Return JSON: {{\"hallucination\": true/false}}"
    )
    structured = llm.with_structured_output(HalluCheck)
    res = structured.invoke(prompt.format(c=context, a=answer))

    state.setdefault("quality", {})
    state["quality"]["hallucination"] = res.hallucination
    return {"quality": state["quality"]}
