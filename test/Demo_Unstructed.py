# UnstructuredFileLoader is a general document loader provided by `langchain_community.document_loaders`
# The underlying layer is based on the [unstructured](https://github.com/Unstructured-IO/unstructured) library.
# It is used to load local files into structured LangChain `Document` objects
# which is convenient for subsequent chunking, embedding, RAG retrieval, etc.
import json
from IPython.core.display import HTML
from IPython.core.display_functions import display
from langchain_unstructured import UnstructuredLoader
import os
from langchain_core.documents import Document


current_dir = os.path.dirname(__file__)
pdf_file = os.path.join(current_dir, "../datas/layout-parser-paper.pdf")

pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]

# Build output directory structure: datas/output/layout-parser-paper/
output_dir = os.path.join(current_dir, "../datas/output", pdf_name)
os.makedirs(output_dir, exist_ok=True)

# Tool function to save JSON
def write_json(data, file_name):
    json_path = os.path.join(output_dir, file_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Explanation of mode parameters:
# "single": The entire file is returned as a Document (no block division)
# "paged": Split by page, each page is a Document
# "elements": Split by structure, each title, paragraph, table, list, etc. is a Document (recommended)

loader = UnstructuredLoader(
    file_path=pdf_file,
    mode="elements",
    strategy="auto",
    coordinates=True,
    partition_via_api=False
)

docs = []
for i, doc in enumerate(loader.lazy_load()):
    docs.append(doc)
    json_name = f"{doc.metadata.get('page_number', 'unknown')}_{i}.json"
    write_json(doc.model_dump(), json_name)


# Output the total number of blocks + information about the first document block
print(f"Num of Docs: {len(docs)}")
print("Metadata of first doc:")
print(docs[0].metadata)
print("Content of first doc:")
print(docs[0].page_content)

print("--" * 50)

# Visualization table (if present)
segments = [
    doc.metadata
    for doc in docs
    if doc.metadata.get("page_number") == 5 and doc.metadata.get("category") == "Table"
]

print("The table paragraph on page 6 reads:")
print(segments)

if segments and "text_as_html" in segments[0]:
    display(HTML(segments[0]["text_as_html"]))

print("--" * 50)



# Read the JSON file just split
def load_doc_from_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return Document(page_content=data['page_content'], metadata=data['metadata'])


example_file = os.path.join(output_dir, "1_3.json")
doc = load_doc_from_json(example_file)
print(doc)








