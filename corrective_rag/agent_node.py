from My_RAG_Project.corrective_rag.graph_state1 import AgentState
from My_RAG_Project.llm_models.embeddings_model import llm
from My_RAG_Project.tools.retriever_tools import retriever_tool
from My_RAG_Project.utils.log_utils import log


def agent_node(state: AgentState):
    """
    The Agent node decides whether to call tools based on the latest user message.
    It binds the retriever tool (PDF RAG) and invokes the LLM on the last message.

    Input:
        state["messages"]: message list

    Output:
        {"messages": [AIMessage or ToolInvocation]} appended to state
    """
    log.info("--- entering agent node ---")
    messages = state["messages"]

    model = llm.bind_tools([retriever_tool])
    response = model.invoke([messages[-1]])
    return {"messages": [response]}
