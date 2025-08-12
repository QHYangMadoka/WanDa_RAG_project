# My_RAG_Project/test/test_milvus_connect.py

import pytest
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from My_RAG_Project.utils.milvus_connect import MilvusLocalController

TEST_COLLECTION_NAME = "test_collection_for_pytest"


@pytest.fixture(scope="module")
def controller():
    return MilvusLocalController()


def test_create_collection(controller):
    client: MilvusClient = controller.client

    if TEST_COLLECTION_NAME in client.list_collections():
        client.drop_collection(TEST_COLLECTION_NAME)

    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4),
        ]
    )

    client.create_collection(TEST_COLLECTION_NAME, schema=schema)
    assert TEST_COLLECTION_NAME in client.list_collections()


def test_insert_sample_data(controller):
    client = controller.client
    entities = [
        {"text": "hello", "embedding": [0.1, 0.2, 0.3, 0.4]},
        {"text": "milvus", "embedding": [0.2, 0.1, 0.5, 0.3]},
        {"text": "pytest", "embedding": [0.3, 0.4, 0.1, 0.2]},
        {"text": "vector", "embedding": [0.0, 0.0, 0.1, 0.9]}
    ]
    result = client.insert(collection_name=TEST_COLLECTION_NAME, data=entities)
    assert "insert_count" in result and result["insert_count"] == 4

def test_describe_collection(controller):
    client = controller.client
    desc = client.describe_collection(TEST_COLLECTION_NAME)
    assert "fields" in desc
    assert any(f["name"] == "embedding" for f in desc["fields"])


def test_search_vector(controller):
    client = controller.client
    client.load_collection(TEST_COLLECTION_NAME)  # 🔧 必须先加载 collection
    results = client.search(
        collection_name=TEST_COLLECTION_NAME,
        data=[[0.1, 0.2, 0.3, 0.4]],
        limit=2,
        search_params={"metric_type": "L2", "params": {"nprobe": 10}},
        output_fields=["text"]
    )
    assert len(results[0]) >= 1
    assert "text" in results[0][0]

def test_drop_collection(controller):
    client = controller.client
    client.drop_collection(TEST_COLLECTION_NAME)
    assert TEST_COLLECTION_NAME not in client.list_collections()
