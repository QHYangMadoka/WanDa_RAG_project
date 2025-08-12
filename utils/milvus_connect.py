import os
from pymilvus import MilvusClient
from My_RAG_Project.utils.env_utils import MILVUS_URI


class MilvusLocalController:
    """Controller for managing local Milvus Standalone and collections"""

    def __init__(self, uri=MILVUS_URI):
        self.client = MilvusClient(uri=uri)

    # ----------- Service Control -----------

    def start_milvus(self):
        """Start local Milvus Standalone service using shell script"""
        os.system("bash standalone_embed.sh start")

    def stop_milvus(self):
        """Stop local Milvus Standalone service"""
        os.system("bash standalone_embed.sh stop")

    def delete_milvus(self):
        """Delete local Milvus containers and persistent data"""
        os.system("bash standalone_embed.sh delete")

    # ----------- Collection Operations -----------

    def list_collections(self):
        """List all existing collections in Milvus"""
        collections = self.client.list_collections()
        print("📂 Existing collections:")
        for name in collections:
            print(" -", name)
        return collections

    def describe_collection(self, collection_name):
        """Describe the schema and metadata of a collection"""
        try:
            desc = self.client.describe_collection(collection_name)
            print(f"🧾 Description of collection '{collection_name}':")
            print(desc)
        except Exception as e:
            print(f"❌ Failed to describe collection: {e}")

    def drop_collection(self, collection_name):
        """Drop a specific collection by name"""
        try:
            self.client.drop_collection(collection_name)
            print(f"✅ Collection '{collection_name}' has been deleted.")
        except Exception as e:
            print(f"❌ Failed to drop collection: {e}")


if __name__ == "__main__":
    controller = MilvusLocalController()

    # Uncomment the lines below to execute operations
    # controller.start_milvus()
    # controller.stop_milvus()
    # controller.delete_milvus()

    controller.list_collections()
    controller.describe_collection("your_collection_name")  # Replace with your collection
    # controller.drop_collection("your_collection_name")
