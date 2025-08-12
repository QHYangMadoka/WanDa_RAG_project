from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from My_RAG_Project.utils.env_utils import OPENAI_API_KEY
from langchain_openai import ChatOpenAI

# Chinese embedding model
bge_model_name = "BAAI/bge-small-zh-v1.5"
bge_model_kwargs = {"device": "cuda"}
bge_encode_kwargs = {"normalize_embeddings": True}

bge_embedding = HuggingFaceEmbeddings(
    model_name=bge_model_name,
    model_kwargs=bge_model_kwargs,
    encode_kwargs=bge_encode_kwargs
)


# English embedding model
openai_embedding = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model="text-embedding-ada-002"
)


llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY
)
