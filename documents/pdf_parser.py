from typing import List
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from My_RAG_Project.llm_models.embeddings_model import openai_embedding
from My_RAG_Project.utils.log_utils import log
from sklearn.feature_extraction.text import TfidfVectorizer


class PDFParser:
    """
    Parser for PDF files. Outputs cleaned and enriched Langchain Document objects
    ready for Milvus ingestion.
    """

    def __init__(self):
        self.semantic_splitter = SemanticChunker(
            openai_embedding,
            breakpoint_threshold_type="standard_deviation",
            breakpoint_threshold_amount=1.0
        )
        self.pre_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
        )

    def parse_pdf(self, file_path: str) -> List[Document]:
        loader = UnstructuredPDFLoader(file_path=file_path, strategy="fast")
        docs = []
        for i, doc in enumerate(loader.lazy_load()):
            content = doc.page_content.strip()
            if not content:
                continue
            doc.metadata["page_number"] = i + 1
            doc.metadata["char_count"] = len(content)
            doc.metadata["source"] = file_path
            docs.append(doc)
        self.add_keywords(docs)
        return docs

    def add_keywords(self, docs: List[Document], top_k: int = 5):
        if not docs:
            return
        contents = [doc.page_content for doc in docs]
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(contents)
        feature_names = vectorizer.get_feature_names_out()

        for i, row in enumerate(X.toarray()):
            top_indices = row.argsort()[::-1][:top_k]
            keywords = [feature_names[j] for j in top_indices if row[j] > 0]
            docs[i].metadata["keywords"] = ", ".join(keywords)

    def text_chunker(self, docs: List[Document]) -> List[Document]:
        chunked = []
        for doc in docs:
            if len(doc.page_content) > 2000:
                # Rough chunk first
                rough_chunks = self.pre_splitter.split_documents([doc])
                for rough in rough_chunks:
                    semantic_chunks = self.semantic_splitter.split_documents([rough])
                    for chunk in semantic_chunks:
                        self._copy_metadata(chunk, doc)
                        chunked.append(chunk)
            else:
                self._copy_metadata(doc, doc)
                chunked.append(doc)
        return chunked

    def _copy_metadata(self, chunk: Document, source_doc: Document):
        chunk.metadata["page_number"] = source_doc.metadata.get("page_number", 0)
        chunk.metadata["char_count"] = len(chunk.page_content)
        chunk.metadata["keywords"] = source_doc.metadata.get("keywords", "")
        chunk.metadata["source"] = source_doc.metadata.get("source", "")

    def parse_pdf_to_documents(self, file_path: str) -> List[Document]:
        raw_docs = self.parse_pdf(file_path)
        log.info(f"Parsed non-empty documents: {len(raw_docs)}")

        chunked_docs = self.text_chunker(raw_docs)
        log.info(f"Semantic chunks after splitting: {len(chunked_docs)}")

        return chunked_docs


if __name__ == '__main__':
    file_path = r'../datas/pdf/(改v2)万达智慧商业逐字稿202409版本 .pdf'
    parser = PDFParser()
    docs = parser.parse_pdf_to_documents(file_path)
    for i, doc in enumerate(docs):
        print(f"[Chunk {i}]")
        print("Metadata:", doc.metadata)
        print("Content:", doc.page_content[:300], "...")
        print("------" * 10)

