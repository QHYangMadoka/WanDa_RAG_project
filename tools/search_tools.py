from typing import List, Optional, Dict, Any
from My_RAG_Project.documents.milvus_db_pdf import MilvusPDFWriter
from My_RAG_Project.utils.env_utils import COLLECTION_NAME, MILVUS_URI
from My_RAG_Project.utils.log_utils import log
from My_RAG_Project.llm_models.embeddings_model import bge_embedding

# PyMilvus low-level imports for sparse/rrf/hybrid
from pymilvus import (
    MilvusClient,
    RRFRanker,
    AnnSearchRequest,
)


# ---------- Common connections ----------

def _get_vectorstore_and_client():
    """Return (vector_store, pymilvus_client) sharing the same collection."""
    mv = MilvusPDFWriter()
    mv.create_connection()  # langchain-milvus vector store
    py_client = MilvusClient(uri=MILVUS_URI)  # low-level client
    return mv.vector_store, py_client


# ---------- 1) Dense similarity with LangChain ----------

def dense_similarity_search(
    query: str,
    k: int = 5,
    expr: Optional[str] = "page_number >= 1",
    output_fields: Optional[List[str]] = None,
):
    """
    Dense vector similarity via LangChain Milvus vector store.
    - query: text query
    - k: top-N
    - expr: Milvus scalar filter (e.g., "page_number >= 1")
    - output_fields: fields to return in metadatas
    """
    vector_store, _ = _get_vectorstore_and_client()
    log.info(f"Dense similarity search: k={k}, expr={expr}")

    # LangChain API: similarity_search returns Documents with .page_content/.metadata
    # We can pass expr through search kwargs via as_retriever (or use similarity_search with filtering if supported).
    # For a quick path, use similarity_search and filter post-hoc if expr is simple; here we rely on retriever for expr.
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "expr": expr} if expr else {"k": k},
    )
    docs = retriever.get_relevant_documents(query)

    rows = []
    for d in docs:
        row = {
            "text": d.page_content,
            "page_number": d.metadata.get("page_number"),
            "keywords": d.metadata.get("keywords"),
            "source": d.metadata.get("source"),
            "char_count": d.metadata.get("char_count"),
        }
        if output_fields:
            row = {k: v for k, v in row.items() if k in output_fields}
        rows.append(row)
    return rows


# ---------- 2) Full-text (BM25) sparse search with PyMilvus ----------

def sparse_bm25_search(
    query: str,
    k: int = 5,
    expr: Optional[str] = "page_number >= 1",
    output_fields: Optional[List[str]] = None,
    search_params: Optional[Dict[str, Any]] = None,
):
    """
    Full-text search (BM25) on 'sparse' field with PyMilvus.
    - query: raw text query; server computes sparse embedding via BM25 function
    - expr: Milvus scalar filter (e.g., "page_number >= 1")
    - search_params: advanced params; defaults are reasonable
    """
    _, client = _get_vectorstore_and_client()
    params = search_params or {
        # You can tune BM25 search params; drop_ratio_search helps skip tiny weights for speed
        "metric_type": "BM25",
        "params": {"drop_ratio_search": 0.2},
    }
    log.info(f"Sparse BM25 search: k={k}, expr={expr}, params={params}")

    res = client.search(
        collection_name=COLLECTION_NAME,
        data=[query],               # raw text; server creates sparse vector via BM25 function
        anns_field="sparse",
        limit=k,
        output_fields=output_fields or ["text", "page_number", "keywords", "source", "char_count"],
        filter=expr or "",
        search_params=params,
    )
    # PyMilvus returns a list of hits per query; we used single query so take res[0]
    hits = res[0] if res else []
    rows = []
    for h in hits:
        row = {
            "text": h.get("text"),
            "page_number": h.get("page_number"),
            "keywords": h.get("keywords"),
            "source": h.get("source"),
            "char_count": h.get("char_count"),
            "_score": h.get("distance"),  # for BM25 higher is better
        }
        rows.append(row)
    return rows


# ---------- 3) Hybrid search (dense + sparse) with RRFRanker ----------

def hybrid_rrf_search(query: str, k: int = 5, rrf_k: int = 60, expr: str = "page_number >= 1"):
    """
    Hybrid search: dense ANN on 'dense' + BM25 on 'sparse', then RRF fuse.
    - query: 用户查询（字符串）
    - k: 返回条数
    - rrf_k: RRF 的平滑参数
    - expr: 过滤表达式
    """
    mv = MilvusPDFWriter()
    mv.create_connection()  # Only establish a connection (do not rebuild the collection)

    # 1) Compute Dense Vector
    dense_vec = bge_embedding.embed_query(query)
    if not isinstance(dense_vec, list) or len(dense_vec) == 0:
        raise ValueError("Dense embedding result is empty.")

    # 2) Dense ANN Search
    dense_res = mv.vector_store.client.search(
        collection_name=COLLECTION_NAME,
        data=[dense_vec],
        anns_field="dense",
        limit=k,
        search_params={"metric_type": "IP", "params": {"ef": 64}},
        filter=expr,
        output_fields=["text", "page_number", "keywords", "source"]
    )

    # 3) Sparse BM25
    sparse_res = mv.vector_store.client.search(
        collection_name=COLLECTION_NAME,
        data=[query],
        anns_field="sparse",
        limit=k,
        search_params={"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}},
        filter=expr,
        output_fields=["text", "page_number", "keywords", "source"]
    )

    # 4) RRF
    def _collect(res_list, key):
        items = []
        for hits in res_list:
            for rank, hit in enumerate(hits):
                doc_id = f"{hit.get('id', '')}_{hit.get('page_number', '')}_{hit.get('source', '')}"
                items.append({
                    "doc_id": doc_id,
                    "rank": rank + 1,
                    "score": float(hit["distance"]) if key == "dense" else float(hit["distance"]),
                    "text": hit.get("text", ""),
                    "page_number": hit.get("page_number", -1),
                    "keywords": hit.get("keywords", ""),
                    "source": hit.get("source", "")
                })
        return items

    dense_items = _collect(dense_res, "dense")
    sparse_items = _collect(sparse_res, "sparse")

    # RRF Compute
    from collections import defaultdict
    fused = defaultdict(lambda: {"score": 0.0, "payload": None})

    def _rrf_score(rank, k=rrf_k):
        return 1.0 / (k + rank)

    for it in dense_items:
        fused[it["doc_id"]]["score"] += _rrf_score(it["rank"], rrf_k)
        fused[it["doc_id"]]["payload"] = it

    for it in sparse_items:
        fused[it["doc_id"]]["score"] += _rrf_score(it["rank"], rrf_k)
        if fused[it["doc_id"]]["payload"] is None:
            fused[it["doc_id"]]["payload"] = it

    # 排序取前 k
    rows = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:k]
    rows = [r["payload"] for r in rows if r["payload"]]

    log.info(f"Hybrid RRF search done. k={k}, rrf_k={rrf_k}, expr={expr}.")
    return rows


# ---------- 4) Simple scalar query (no vectors) ----------

def scalar_query(
    expr: str,
    limit: int = 10,
    output_fields: Optional[List[str]] = None,
):
    """
    Scalar-only query via PyMilvus (no vector computation).
    """
    _, client = _get_vectorstore_and_client()
    rows = client.query(
        collection_name=COLLECTION_NAME,
        filter=expr,  # e.g., "page_number == 1 and char_count > 500"
        output_fields=output_fields or ["text", "page_number", "keywords", "source", "char_count"],
        limit=limit,
    )
    return rows


# ---------- Demo main ----------

if __name__ == "__main__":
    q = "万达智慧商业 平台 增值服务 是什么？"

    print("\n=== Dense similarity ===")
    rows = dense_similarity_search(q, k=3)
    for i, r in enumerate(rows):
        print(f"[{i}] p{r.get('page_number')} | kw={r.get('keywords')} | {r.get('text','')[:80]}...")

    print("\n=== Sparse BM25 ===")
    rows = sparse_bm25_search(q, k=3)
    for i, r in enumerate(rows):
        print(f"[{i}] p{r.get('page_number')} | kw={r.get('keywords')} | {r.get('text','')[:80]}...")

    print("\n=== Hybrid RRF ===")
    rows = hybrid_rrf_search(q, k=5, rrf_k=60)
    for i, r in enumerate(rows):
        print(f"[{i}] p{r.get('page_number')} | kw={r.get('keywords')} | {r.get('text','')[:80]}...")

    print("\n=== Scalar query (page 1) ===")
    rows = scalar_query("page_number == 1", limit=5)
    for i, r in enumerate(rows):
        print(f"[{i}] p{r.get('page_number')} | kw={r.get('keywords')} | {r.get('text','')[:80]}...")
