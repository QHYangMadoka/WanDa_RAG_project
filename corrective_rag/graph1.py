import uuid
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from My_RAG_Project.corrective_rag.agent_node import agent_node
from My_RAG_Project.corrective_rag.generate_node import generate
from My_RAG_Project.corrective_rag.get_human_message import get_last_human_message
from My_RAG_Project.corrective_rag.graph_state1 import AgentState, Grade
from My_RAG_Project.corrective_rag.rewrite_node import rewrite
from My_RAG_Project.llm_models.embeddings_model import llm
from My_RAG_Project.tools.retriever_tools import retriever_tool
from My_RAG_Project.utils.log_utils import log
from My_RAG_Project.utils.print_utils import _print_event  # keep your existing printer


def _stringify_tool_output_for_grade(docs) -> str:
    """
    Normalize tool output for the grading step.
    If `docs` is a list of Documents, join their content with page/keywords markers.
    """
    try:
        from langchain_core.documents import Document
    except Exception:
        Document = None

    if isinstance(docs, list) and docs:
        parts = []
        for d in docs:
            content = getattr(d, "page_content", None) or str(d)
            meta = getattr(d, "metadata", {}) or {}
            page = meta.get("page_number", "")
            kw = meta.get("keywords", "")
            parts.append(f"[p{page} | kw={kw}]\n{content}")
        return "\n\n---\n\n".join(parts)
    if isinstance(docs, str):
        return docs
    return str(docs)


def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Judge whether retrieved documents are relevant to the user query.
    Returns 'generate' if relevant, else 'rewrite'.
    """
    log.info("--- grading retrieved docs ---")

    llm_with_structured = llm.with_structured_output(Grade)

    prompt = PromptTemplate(
        template=(
            "You are a grader of retrieval relevance.\n\n"
            "Retrieved context:\n{context}\n\n"
            "User question:\n{question}\n\n"
            "If the context contains keywords or semantic content relevant to the question, "
            "respond with binary_score = 'yes', otherwise 'no'."
        ),
        input_variables=["context", "question"],
    )
    chain = prompt | llm_with_structured

    messages = state["messages"]
    last_message = messages[-1]
    question = get_last_human_message(messages).content

    docs = getattr(last_message, "content", "")
    context = _stringify_tool_output_for_grade(docs)

    scored = chain.invoke({"question": question, "context": context})
    score = scored.binary_score.strip().lower()

    if score == "yes":
        print("--- result: relevant ---")
        return "generate"
    else:
        print("--- result: not relevant ---")
        return "rewrite"


# Build the LangGraph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "retrieve", END: END},
)
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("rewrite", "agent")
workflow.add_edge("generate", END)

# Memory/checkpoint
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

_printed = set()

if __name__ == "__main__":
    while True:
        question = input("用户：")
        if question.lower() in ["q", "exit", "quit"]:
            log.info("对话结束，拜拜！")
            break
        inputs = {"messages": [("user", question)]}
        events = graph.stream(inputs, config=config, stream_mode="values")
        for event in events:
            _print_event(event, _printed)
