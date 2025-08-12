import os
import sys
import time
import glob
import queue
import multiprocessing as mp
from typing import List

from My_RAG_Project.utils.log_utils import log
from My_RAG_Project.utils.env_utils import MILVUS_URI
from pymilvus import MilvusClient
from pymilvus.client.types import DataType, MetricType
from pymilvus import IndexType, Function
from pymilvus.client.types import FunctionType

# --------------- Parser process ---------------

def file_parser_process(pdf_dir: str, output_queue: mp.Queue, batch_size: int = 20):
    """
    Process-1: Parse all PDFs under a directory and put chunked docs into a queue by batch.
    """
    from My_RAG_Project.documents.pdf_parser import PDFParser  # safe (no CUDA init)
    log.info(f"Parser process scanning dir: {pdf_dir}")

    pdf_paths = sorted(
        [p for p in glob.glob(os.path.join(pdf_dir, "*")) if p.lower().endswith(".pdf")]
    )
    if not pdf_paths:
        log.warning("No PDF files found in the directory.")
        output_queue.put(None)
        return

    parser = PDFParser()
    buffer: List = []
    total_chunks = 0

    for pdf in pdf_paths:
        try:
            docs = parser.parse_pdf_to_documents(pdf)
            if docs:
                buffer.extend(docs)
                total_chunks += len(docs)

            if len(buffer) >= batch_size:
                # push a batch
                output_queue.put(buffer.copy())
                buffer.clear()
        except Exception as e:
            log.error(f"Failed to parse {pdf}: {e}", exc_info=True)

    if buffer:
        output_queue.put(buffer)

    output_queue.put(None)
    log.info(f"Parser process finished. Parsed {len(pdf_paths)} PDFs, total chunks: {total_chunks}")


# --------------- Writer process ---------------

def milvus_writer_process(input_queue: mp.Queue, collection_name: str, milvus_uri: str):
    """
    Process-2: Initialize embedding and Milvus vector store in THIS process and write batches.
    This is where CUDA can be safely initialized (spawn).
    """
    # Lazy import here so CUDA init happens in the child process, not parent.
    from langchain_milvus import Milvus, BM25BuiltInFunction
    from My_RAG_Project.llm_models.embeddings_model import bge_embedding

    # Build the vectorstore connection in the writer process
    vector_store = Milvus(
        embedding_function=bge_embedding,                # CUDA/HF init happens here
        collection_name=collection_name,
        builtin_function=BM25BuiltInFunction(),
        vector_field=["dense", "sparse"],
        consistency_level="Strong",
        auto_id=True,
        connection_args={"uri": milvus_uri},
    )

    total_written = 0
    while True:
        try:
            batch = input_queue.get()
            if batch is None:
                break
            if not batch:
                continue

            vector_store.add_documents(batch)
            total_written += len(batch)
            log.info(f"Written batch. Total written: {total_written}")
        except Exception as e:
            log.error(f"Written failed: {e}", exc_info=True)

    log.info(f"The writing process has ended. A total of {total_written} documents have been written.")


# --------------- Collection utilities ---------------

def collection_exists(client: MilvusClient, collection_name: str) -> bool:
    return collection_name in client.list_collections()

def drop_collection_if_exists(client: MilvusClient, collection_name: str):
    if collection_exists(client, collection_name):
        # release -> drop indexes -> drop collection (defensive)
        try:
            client.release_collection(collection_name)
        except Exception:
            pass
        try:
            for idx in client.list_indexes(collection_name):
                client.drop_index(collection_name, index_name=idx)
        except Exception:
            pass
        client.drop_collection(collection_name)

def create_pdf_collection(client: MilvusClient, collection_name: str):
    """
    Create a collection compatible with pdf_parser & milvus_db_pdf:
      - text (VARCHAR, analyzer enabled), source, page_number, char_count, keywords
      - sparse (SPARSE_FLOAT_VECTOR) + dense (FLOAT_VECTOR dim=512)
      - BM25 function on text -> sparse
      - HNSW index on dense; SPARSE_INVERTED_INDEX on sparse
    """
    schema = client.create_schema()
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=10000,
        enable_analyzer=True,
        analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]},
    )
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="page_number", datatype=DataType.INT64)
    schema.add_field(field_name="char_count", datatype=DataType.INT64)
    schema.add_field(field_name="keywords", datatype=DataType.VARCHAR, max_length=2000)

    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=512)

    bm25_func = Function(
        name="text_bm25_emb",
        input_field_names=["text"],
        output_field_names=["sparse"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_func)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="sparse",
        index_name="sparse_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
    )
    index_params.add_index(
        field_name="dense",
        index_name="dense_index",
        index_type=IndexType.HNSW,
        metric_type=MetricType.IP,
        params={"M": 16, "efConstruction": 64},
    )

    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    log.info(f"✅ Created Milvus collection: {collection_name}")


def prepare_collection_interactive(client: MilvusClient) -> str:
    """
    Ask user for collection name.
    - If exists: ask whether to drop and recreate.
    - If 'n' or anything else: exit program.
    """
    name = input("Please input the Collection name: ").strip()
    import re
    if not re.match(r'^[A-Za-z0-9_]+$', name):
        print("❌ Collection name can only contain letters, numbers, and underscores.")
        exit()

    if not name:
        print("Empty collection name, exit.")
        sys.exit(0)

    if collection_exists(client, name):
        ans = input(f"Collection '{name}' already exists. Drop and recreate? (y/N): ").strip().lower()
        if ans == "y":
            drop_collection_if_exists(client, name)
            create_pdf_collection(client, name)
        else:
            print("You chose not to drop the existing collection. Exit.")
            sys.exit(0)
    else:
        create_pdf_collection(client, name)

    return name


# --------------- Main ---------------

def main():
    pdf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datas", "pdf"))
    queue_maxsize = 20
    batch_size = 20

    # Prepare collection (parent process, no CUDA touched)
    client = MilvusClient(uri=MILVUS_URI)
    collection_name = prepare_collection_interactive(client)

    # Use spawn context for safety with CUDA
    ctx = mp.get_context("spawn")
    docs_queue: mp.Queue = ctx.Queue(maxsize=queue_maxsize)

    # Start processes
    parser_proc = ctx.Process(
        target=file_parser_process, args=(pdf_dir, docs_queue, batch_size), name="parser-proc"
    )
    writer_proc = ctx.Process(
        target=milvus_writer_process, args=(docs_queue, collection_name, MILVUS_URI), name="writer-proc"
    )

    parser_proc.start()
    writer_proc.start()

    parser_proc.join()
    docs_queue.put(None)  # ensure writer can exit
    writer_proc.join()

    # Wait for rows to be visible and then print stats & sample rows
    time.sleep(0.3)
    # Some engines are async; do a lightweight retry for row_count
    for _ in range(10):
        try:
            stats = client.get_collection_stats(collection_name)
            row_count = stats.get("row_count", 0)
            if int(row_count) > 0:
                break
            time.sleep(0.3)
        except Exception:
            time.sleep(0.3)

    stats = client.get_collection_stats(collection_name)
    row_count = int(stats.get("row_count", 0))
    print(f"📊 Total rows in collection: {row_count}")

    # Print first 10 docs
    try:
        res = client.query(
            collection_name=collection_name,
            filter="",                # empty filter must be paired with limit
            limit=10,
            output_fields=["text", "page_number", "keywords", "source", "char_count"],
        )
        print("🔍 Query Result (first 10):")
        for i, r in enumerate(res):
            page = r.get("page_number", None)
            kws = r.get("keywords", "")
            txt = r.get("text", "") or ""
            print(f"[Doc {i}] Page {page} - Keywords: {kws}")
            print(txt[:200] + ("..." if len(txt) > 200 else ""))
            print("-" * 60)
    except Exception as e:
        log.error(f"Query failed: {e}", exc_info=True)


if __name__ == "__main__":
    # MUST set spawn BEFORE creating any processes
    mp.set_start_method("spawn", force=True)
    main()
