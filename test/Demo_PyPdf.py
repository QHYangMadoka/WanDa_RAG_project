from langchain_community.document_loaders import PyPDFLoader
import os
# Load a PDF file and parse each page into a separate Document object
# Prepare for vectorization, storage, etc. in subsequent RAG projects

current_dir = os.path.dirname(__file__)
pdf_file = os.path.join(current_dir, "../datas/layout-parser-paper.pdf")

# Create a PyPDFLoader instance to load this PDF
loader = PyPDFLoader(file_path=pdf_file)
docs = loader.load()


print(f'Number of docs: {len(docs)}')
print(docs[0].metadata)
print(docs[0].page_content)
print('-' * 50)