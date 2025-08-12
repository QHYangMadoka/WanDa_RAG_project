from My_RAG_Project.utils.log_utils import log
from My_RAG_Project.llm_models.embeddings_model import llm
from langchain_core.messages import HumanMessage


def transform_query_node(state):
    """
    Light-weight query refinement to improve retrieval hit rate.
    - Keep domain keywords.
    - Expand synonyms or add constraints when needed.
    """
    log.info("[Adaptive] transform_query_node")
    original = state.get("query") or state.get("user_input") or ""
    if not original.strip():
        return {}

    # Simple heuristic: if query is too short, ask LLM to expand with more keywords.
    if len(original.strip()) < 6:
        prompt = HumanMessage(
            content=(
                "Rewrite a short Chinese query into a retrieval-friendly form. "
                "Keep domain keywords and add 3~5 relevant terms. "
                f"Original: {original}\n"
                "Return only the query; do not add explanations."
            )
        )
        refined = llm.invoke([prompt]).content.strip()
        if refined:
            state["query"] = refined
            return {"query": refined}

    # Otherwise keep as is
    state["query"] = original
    return {"query": original}
