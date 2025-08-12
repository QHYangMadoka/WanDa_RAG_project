from typing import List
from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import IndexType, MilvusClient, Function
from pymilvus.client.types import MetricType, DataType, FunctionType

from My_RAG_Project.documents.pdf_parser import PDFParser
from My_RAG_Project.llm_models.embeddings_model import bge_embedding
from My_RAG_Project.utils.env_utils import MILVUS_URI, COLLECTION_NAME
from My_RAG_Project.utils.log_utils import log


def validate_docs(docs: List[Document]):
    for i, doc in enumerate(docs):
        if len(doc.page_content) > 10000:
            raise ValueError(f"Doc {i} exceeds 10000 characters")
        if not isinstance(doc.metadata.get("keywords", ""), str):
            raise TypeError(f"Doc {i} has invalid 'keywords': {doc.metadata.get('keywords')}")
        if not isinstance(doc.metadata.get("page_number", 0), int):
            raise TypeError(f"Doc {i} has invalid 'page_number'")
        if not isinstance(doc.metadata.get("char_count", 0), int):
            raise TypeError(f"Doc {i} has invalid 'char_count'")


class MilvusPDFWriter:
    """Write parsed PDF documents into Milvus"""

    def __init__(self):
        self.vector_store: Milvus = None

    def create_collection(self):
        client = MilvusClient(uri=MILVUS_URI)
        schema = client.create_schema()
        schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=10000, enable_analyzer=True,
                         analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]})
        schema.add_field(field_name='source', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='page_number', datatype=DataType.INT64)
        schema.add_field(field_name='char_count', datatype=DataType.INT64)
        schema.add_field(field_name='keywords', datatype=DataType.VARCHAR, max_length=2000)
        schema.add_field(field_name='sparse', datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name='dense', datatype=DataType.FLOAT_VECTOR, dim=512)

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

        if COLLECTION_NAME in client.list_collections():
            client.release_collection(COLLECTION_NAME)
            client.drop_index(COLLECTION_NAME, index_name='sparse_index')
            client.drop_index(COLLECTION_NAME, index_name='dense_index')
            client.drop_collection(COLLECTION_NAME)

        client.create_collection(COLLECTION_NAME, schema=schema, index_params=index_params)
        log.info(f"✅ Created Milvus collection: {COLLECTION_NAME}")

    def create_connection(self):
        self.vector_store = Milvus(
            embedding_function=bge_embedding,
            collection_name=COLLECTION_NAME,
            builtin_function=BM25BuiltInFunction(),
            vector_field=['dense', 'sparse'],
            consistency_level="Strong",
            auto_id=True,
            connection_args={"uri": MILVUS_URI}
        )
        log.info("🔗 Connected to Milvus with embedding and BM25 support")

    def add_documents(self, docs: List[Document]):
        try:
            validate_docs(docs)
            self.vector_store.add_documents(docs)
            log.info(f"📄 Added {len(docs)} documents to Milvus collection")
        except Exception as e:
            log.error(f"❌ Failed to add documents to Milvus: {e}")
            for i, doc in enumerate(docs):
                text = doc.page_content if doc.page_content else ""
                log.warning(f"Doc {i} failed:\nMeta: {doc.metadata}\nText Preview: {text[:100]}...")


if __name__ == '__main__':
    # Example test run
    file_path = r'../datas/pdf/(改v2)万达智慧商业逐字稿202409版本 .pdf'
    parser = PDFParser()
    docs = parser.parse_pdf_to_documents(file_path)

    milvus = MilvusPDFWriter()
    milvus.create_collection()
    milvus.create_connection()

    # Insert
    milvus.add_documents(docs)

    # --- Make data visible for query/search: flush + load + wait (important) ---
    client = milvus.vector_store.client

    # 1) Flush to seal segments so row_count updates
    client.flush(collection_name=COLLECTION_NAME)
    # 2) Load collection to enable query/search
    client.load_collection(collection_name=COLLECTION_NAME)

    # 3) Wait until row_count reflects the newly inserted rows
    import time
    expected = len(docs)
    for _ in range(30):  # up to ~15s (30 * 0.5s)
        stats = client.get_collection_stats(COLLECTION_NAME)
        row_count = int(stats.get("row_count", 0))
        print(f"⏳ Waiting rows... row_count={row_count}", flush=True)
        if row_count >= expected:
            break
        time.sleep(0.5)

    # Print final row_count
    stats = client.get_collection_stats(COLLECTION_NAME)
    total_rows = int(stats.get("row_count", 0))
    print(f"📊 Total rows in collection: {total_rows}", flush=True)

    # 4) Query (note: empty filter requires a limit)
    result = client.query(
        collection_name=COLLECTION_NAME,
        filter="",  # empty expr allowed with limit
        output_fields=['text', 'page_number', 'keywords'],
        limit=20
    )

    print("🔍 Query Result:")
    for i, r in enumerate(result):
        page = r.get("page_number", "N/A")
        kws = r.get("keywords", "")
        txt = (r.get("text") or "")[:200]
        print(f"[Doc {i}] Page {page} - Keywords: {kws}\n{txt}...\n", flush=True)