# retriever_tools.py
from langchain_core.tools import create_retriever_tool
from My_RAG_Project.documents.milvus_db_pdf import MilvusPDFWriter
from My_RAG_Project.utils.env_utils import COLLECTION_NAME

# Initialize a Milvus-backed vector store that matches the PDF schema.
# We only connect to an existing collection; creation is done elsewhere.
mv = MilvusPDFWriter()
mv.create_connection()  # Uses MILVUS_URI and COLLECTION_NAME from env

# Build a Retriever from the Milvus vector store.
# NOTE:
# - search_type="similarity" runs dense vector similarity by default.
# - For scalar filtering in langchain-milvus, prefer "expr" with Milvus syntax.
retriever = mv.vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,
        # Use expr to filter via Milvus scalar expression language.
        # Adapt to your schema: page_number >= 1 selects all valid chunks.
        "expr": "page_number >= 1",
        # Optional: if you want to drop very weak matches, uncomment:
        # "score_threshold": 0.1,
        # Optional RRF in some builds of langchain-milvus; keep commented if not needed:
        # "ranker_type": "rrf", "ranker_params": {"k": 100},
    },
)

# Expose a tool for agent-style usage
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="pdf_rag_retriever",
    description=(
        "Retrieve relevant chunks from the PDF collection. "
        f"Milvus collection: {COLLECTION_NAME}. "
        "Fields: text, source, page_number, char_count, keywords."
    ),
)

__all__ = ["retriever_tool", "retriever"]
