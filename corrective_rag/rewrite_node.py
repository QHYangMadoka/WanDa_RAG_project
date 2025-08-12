from langchain_core.messages import HumanMessage

from My_RAG_Project.corrective_rag.get_human_message import get_last_human_message
from My_RAG_Project.llm_models.embeddings_model import llm
from My_RAG_Project.utils.log_utils import log


def rewrite(state):
    """
    Rewrite the user's query to improve retrieval.
    Returns an AIMessage with a refined query proposal.
    """
    log.info("--- rewriting query ---")
    messages = state["messages"]
    question = get_last_human_message(messages).content

    msg = [
        HumanMessage(
            content=(
                "Analyze the input and infer the underlying intent.\n"
                "This is the original question:\n"
                "-------\n"
                f"{question}\n"
                "-------\n"
                "Please propose a better, more retrieval-friendly query:"
            )
        )
    ]
    response = llm.invoke(msg)
    return {"messages": [response]}
