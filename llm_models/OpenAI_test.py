from openai import OpenAI
from My_RAG_Project.utils.env_utils import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=["test embedding"]
)

print(response.data[0].embedding[:10])