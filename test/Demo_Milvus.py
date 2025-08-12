# 导入必要的库
from pymilvus import MilvusClient  # Milvus 客户端，用于操作嵌入式向量数据库
import numpy as np  # NumPy 库，用于生成随机数和处理数组

# 初始化 Milvus 客户端
# 参数 "./milvus_demo.db" 指定了本地数据库文件的路径
# client = MilvusClient("./milvus_demo.db")
# 如果该路径下的 milvus_demo.db 文件不存在，Milvus Lite 会自动：
# 创建这个 .db 文件；
# 初始化必要的元数据表结构；
# 模拟一个本地向量数据库环境。

def test_milvus_workflow():
    client = MilvusClient("./milvus_demo.db")

    if "demo_collection" in client.list_collections():
        client.drop_collection(collection_name="demo_collection")

    # Create a collection
    # A collection is similar to a table in a relational database, used to store vectors and other fields
    client.create_collection(collection_name="demo_collection", dimension=384)

    # Prepare data: documents, vectors, and other fields
    docs = [
        "Artificial intelligence was founded as an academic discipline in 1956.",
        "Alan Turing was the first person to conduct substantial research in AI.",
        "Born in Maida Vale, London, Turing was raised in southern England.",
    ]

    # Generate a random 384-dimensional vector for each paragraph of text
    # Generate random numbers between -1 and 1 using NumPy's `np.random.uniform`
    vectors = [[np.random.uniform(-1, 1) for _ in range(384)] for _ in range(len(docs))]
    data = [{"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"} for i in range(len(vectors))]

    res = client.insert(collection_name="demo_collection", data=data)
    # Output the insertion result (usually returns a success status or the number of records inserted)
    print("Insert result:", res)

    # Pack documents, vectors, IDs and subjects into a dictionary format
    # Each dictionary contains the following fields:
    # - id: unique identifier
    # - vector: vector data
    # - text: text content
    # - subject: subject label (here "history")
    res = client.search(
        collection_name="demo_collection",
        data=[vectors[0]],
        filter="subject == 'history'",
        limit=2,
        output_fields=["text", "subject"]
    )


    print("Search result:", res)

    res = client.query(
        collection_name="demo_collection",
        filter="subject == 'history'",
        output_fields=["text", "subject"]
    )
    print("Query result:", res)

    res = client.delete(collection_name="demo_collection", filter="subject == 'history'")
    print("Delete result:", res)


if __name__ == "__main__":
    test_milvus_workflow()