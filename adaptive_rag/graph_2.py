import uuid
from langgraph.graph import StateGraph
from langgraph.constants import START, END
from langgraph.checkpoint.memory import MemorySaver

from My_RAG_Project.utils.log_utils import log
from My_RAG_Project.adaptive_rag.graph_state2 import AdaptiveState, default_retrieval_params, inc_iterations
from My_RAG_Project.adaptive_rag.transform_query_node import transform_query_node
from My_RAG_Project.adaptive_rag.query_route_chain import query_route_chain
from My_RAG_Project.adaptive_rag.retriever_node import retriever_node
from My_RAG_Project.adaptive_rag.grade_documents_node import grade_documents_node
from My_RAG_Project.adaptive_rag.web_search_node import web_search_node
from My_RAG_Project.adaptive_rag.generate_node2 import generate_node2
from My_RAG_Project.adaptive_rag.grade_answer_chain import grade_answer_chain
from My_RAG_Project.adaptive_rag.grade_hallucinations_chain import grade_hallucinations_chain


def _route_after_docs(state: AdaptiveState):
    """
    Decide next hop after grading documents.
    - Need more docs? then go transform_query (iterate) or web if flagged.
    - Otherwise generate.
    """
    if state.get("need_more_docs"):
        # Stop if reached iteration limit
        if inc_iterations(state, max_iters=3):
            return "generate"
        if state.get("needs_web"):
            return "web_search"
        return "transform_query"
    return "generate"


def _route_after_answer(state: AdaptiveState):
    """
    If answer failed quality check, try another retrieval round.
    """
    if state.get("need_retry_answer"):
        if inc_iterations(state, max_iters=3):
            return END
        return "transform_query"
    return "hallucination_check"


def build_graph():
    g = StateGraph(AdaptiveState)

    # Nodes
    g.add_node("transform_query", transform_query_node)
    g.add_node("query_route", query_route_chain)
    g.add_node("retriever", retriever_node)
    g.add_node("grade_docs", grade_documents_node)
    g.add_node("web_search", web_search_node)
    g.add_node("generate", generate_node2)
    g.add_node("answer_grade", grade_answer_chain)
    g.add_node("hallucination_check", grade_hallucinations_chain)

    # Edges
    g.add_edge(START, "transform_query")
    g.add_edge("transform_query", "query_route")
    g.add_edge("query_route", "retriever")
    g.add_edge("retriever", "grade_docs")
    g.add_conditional_edges("grade_docs", _route_after_docs,
                            {"transform_query": "transform_query",
                             "web_search": "web_search",
                             "generate": "generate"})
    g.add_edge("web_search", "grade_docs")
    g.add_edge("generate", "answer_grade")
    g.add_conditional_edges("answer_grade", _route_after_answer,
                            {"transform_query": "transform_query",
                             "hallucination_check": "hallucination_check",
                             END: END})
    g.add_edge("hallucination_check", END)

    memory = MemorySaver()
    return g.compile(checkpointer=memory)


if __name__ == "__main__":
    log.info("[Adaptive] Graph v2 starting...")
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    while True:
        q = input("用户> ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            break
        state = {
            "messages": [],
            "user_input": q,
            "query": q,
            "retrieval_params": default_retrieval_params(),
            "docs": [],
            "filtered_docs": [],
            "answer": "",
            "citations": [],
            "needs_web": False,
            "quality": {},
            "iterations": 0,
        }
        for ev in graph.stream(state, config=config, stream_mode="values"):
            pass

        print("\n=== Answer ===")
        print(graph.get_state(config)["answer"])
        print("\n=== Citations (top) ===")
        for c in graph.get_state(config).get("citations", [])[:5]:
            print(f"- p{c['page_number']} | score={c['score']} | {c['source']} | {c['snippet'][:80]}...")
        print()
