import json
from IPython.core.display import HTML
from IPython.core.display_functions import display
from langchain_unstructured import UnstructuredLoader
import os
from langchain_core.documents import Document

loader = UnstructuredLoader(
    file_path=r'../datas/md/operational_faq.md',
    mode='elements',
    strategy='fast',              # markdown 建议用 fast
    partition_via_api=False       # 用本地方式解析
)

docs = loader.load()
print(f'Num of Docs: {len(docs)}')

for i in range(min(10, len(docs))):
    print(docs[i].metadata)
    print(docs[i].page_content)
    print('--' * 50)

