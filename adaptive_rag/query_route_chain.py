from My_RAG_Project.utils.log_utils import log
from My_RAG_Project.adaptive_rag.graph_state2 import default_retrieval_params


def query_route_chain(state):
    """
    Decide retrieval strategy and whether to use web search.
    Heuristics:
    - If query contains open-world intent words, set needs_web=True.
    - Otherwise use hybrid by default with expr and k as given.
    """
    log.info("[Adaptive] query_route_chain")
    query = state.get("query") or state.get("user_input") or ""
    params = state.get("retrieval_params") or default_retrieval_params()

    # Very simple open-world detector
    open_world_signals = ["最新", "新闻", "什么时候发布", "外部", "官网", "价格", "对比", "开源"]
    needs_web = any(s in query for s in open_world_signals)

    # Hybrid default; you can set rules to switch
    params["strategy"] = "hybrid"
    params.setdefault("k", 5)
    params.setdefault("rrf_k", 60)
    params.setdefault("expr", "page_number >= 1")

    state["retrieval_params"] = params
    state["needs_web"] = needs_web
    return {"retrieval_params": params, "needs_web": needs_web}
