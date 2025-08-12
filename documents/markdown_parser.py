from typing import List
from langchain_experimental.text_splitter import SemanticChunker

from My_RAG_Project.llm_models.embeddings_model import openai_embedding
from My_RAG_Project.utils.log_utils import log
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document


class MarkdownParser:
    """
    responsible for parsing and slicing markdown files
    """
    def __init__(self):
        self.text_splitter = SemanticChunker(
            openai_embedding, breakpoint_threshold_type="percentile"
        )

    def text_chunker(self, datas: List[Document]) -> List[Document]:
        new_docs = []
        for d in datas:
            # If the content exceeds the threshold, it will be segmented according to semantics
            if len(d.page_content) > 3000:
                new_docs.extend(self.text_splitter.split_documents([d]))
                continue
            new_docs.append(d)
        return new_docs


    def parse_markdown_to_documents(self, md_file: str, encoding='utf-8') -> List[Document]:
        documents = self.parse_markdown(md_file)
        log.info(f'Docs length after parsing: {len(documents)}')

        merged_documents = self.merge_title_content(documents)

        log.info(f'The length of the merged documents: {len(merged_documents)}')

        chunk_documents = self.text_chunker(merged_documents)
        log.info(f'Length after semantic cutting: {len(chunk_documents)}')
        return chunk_documents

    def parse_markdown(self, md_file: str) -> List[Document]:
        loader = UnstructuredMarkdownLoader(
            file_path=md_file,
            mode='elements',
            strategy='fast'
        )
        docs = []
        for doc in loader.lazy_load():
            docs.append(doc)

        return docs

    def merge_title_content(self, datas: List[Document]) -> List[Document]:
        merged_data = []
        parent_dict = {}  # stores all parent documents, key is the ID of the current parent document.
        for document in datas:
            metadata = document.metadata
            if 'languages' in metadata:
                metadata.pop('languages')

            parent_id = metadata.get('parent_id', None)
            category = metadata.get('category', None)
            element_id = metadata.get('element_id', None)

            if category == 'NarrativeText' and parent_id is None:  # is content document
                merged_data.append(document)
            if category == 'Title':
                document.metadata['title'] = document.page_content
                if parent_id in parent_dict:
                    document.page_content = parent_dict[parent_id].page_content + ' -> ' + document.page_content
                parent_dict[element_id] = document
            if category != 'Title' and parent_id:
                parent_dict[parent_id].page_content = parent_dict[parent_id].page_content + ' ' + document.page_content
                parent_dict[parent_id].metadata['category'] = 'content'

        if parent_dict is not None:
            merged_data.extend(parent_dict.values())

        return merged_data


if __name__ == '__main__':
    file_path = r'../datas/md/operational_faq.md'
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)
    for item in docs:
        print(f"MetaData: {item.metadata}")
        print(f"Title: {item.metadata.get('title', None)}")
        print(f"Content of Doc: {item.page_content}\n")
        print("------" * 10)
