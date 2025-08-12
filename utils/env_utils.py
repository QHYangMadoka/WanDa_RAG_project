import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

MILVUS_URI = 'http://172.25.112.1:19530'

COLLECTION_NAME = 'wanda_commerce'
