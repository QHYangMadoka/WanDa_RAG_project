# WanDa_RAG_project

### 📌 Project Background

This project was built during my internship at [AI Indeed (实在智能)](https://www.ai-indeed.com/), where I had the opportunity to participate in the development of a knowledge-based RAG (Retrieval-Augmented Generation) system. The system uses test data from the **Wanda RAG project** I encountered during my internship. Inspired by that experience, I independently designed and implemented this RAG demo system from scratch, applying the knowledge I learned throughout the internship.

---

### ⚙️ Tech Stack

The project is implemented in Python and built on top of the following major components:

- **Milvus Standalone**: high-performance vector database (deployed via Docker)
- **LangChain**: modular framework for building LLM-powered applications
- **LangGraph**: agent workflow orchestration (experimental/optional)
- **Unstructured**: local parsing of PDFs, Markdown, and Office documents
- **BM25 + Dense Embeddings**: hybrid search combining keyword and semantic retrieval
- **pymilvus**: official Python client for Milvus
- **sentence-transformers / HuggingFace**: for generating dense vector embeddings
- **OpenAI or local LLMs**: for generation tasks (via LangChain interface)

---

### 🚀 Milvus Setup on Ubuntu via Docker Desktop (WSL2)

This project requires Milvus Standalone running in a **Ubuntu WSL2 environment** using **Docker Desktop** on Windows.

#### ✅ Prerequisites

- Windows 10/11 with **WSL2 enabled**
- **Docker Desktop for Windows** installed
- WSL Integration enabled for your Ubuntu distribution in Docker Desktop settings


### 📂 Project Structure


My_RAG_Project/
│
├── agent/ # RAG agent and LangGraph workflow
│ ├── rag_agent.py # Main RAG agent logic (PDF-adapted)
│ ├── graph_2.py # Adaptive/Corrective RAG graph structure
│ └── ... # Nodes and chains for grading, query routing, etc.
│
├── documents/
│ ├── pdf_parser.py # PDF parsing & semantic chunking
│ ├── milvus_db_pdf.py # Milvus connection and schema for PDF docs
│ ├── write_milvus_pdf.py # Multi-process PDF ingestion into Milvus
│
├── tools/
│ ├── search_tools.py # Dense, sparse, hybrid retrieval tools (PDF-adapted)
│ ├── retriever_tools.py # Retriever wrappers for Milvus
│
├── utils/
│ ├── log_utils.py # Logging configuration
│ ├── env_utils.py # Environment variable loading (keys, configs

### 📄 PDF Parsing & Storage Workflow

#### PDF Parsing (`pdf_parser.py`)

- Uses **unstructured** to extract text content from PDF files.  
- Performs **semantic chunking** with configurable thresholds (optimized for Chinese/English mixed text).  
- Extracts metadata: `source`, `page_number`, `char_count`, `keywords`.  

#### Milvus Storage (`milvus_db_pdf.py` + `write_milvus_pdf.py`)

- Creates a Milvus collection with fields:  
  - `text` (content), `source`, `page_number`, `char_count`, `keywords`  
  - `dense` (FloatVector, from embeddings)  
  - `sparse` (SparseFloatVector, from BM25)  
- Supports **multi-process ingestion**: one process for parsing, one for writing.  
- Automatically deletes/recreates collection if name is taken.  

#### Hybrid Indexing

- **Dense index**: HNSW, Inner Product similarity  
- **Sparse index**: BM25  
- Enables **hybrid RRF (Reciprocal Rank Fusion) retrieval**.  

---

### 🔍 Retrieval Strategies

Implemented in `tools/search_tools.py`:  

- **Dense Similarity Search**  
  - Embeds queries using the same model as document embeddings.  
  - Retrieves top-K semantic matches.  

- **Sparse BM25 Search**  
  - Token-based retrieval optimized for keyword precision.  

- **Hybrid RRF Search**  
  - Combines dense and sparse rankings for balanced precision/recall.  

- **Scalar Filtering**  
  - Example: `expr="page_number >= 2"` for field-based filtering.  

Each search method returns **document text + metadata + score** for transparency.  


### 🛠️ Corrective RAG

Corrective RAG is designed to **detect and fix low-quality retrievals** before final answer generation.  
It introduces a **grading layer** that evaluates retrieved documents and answers.  

#### Components

- **Retriever Node (`retriever_node.py`)**  
  Fetches candidate documents from Milvus using dense, sparse, or hybrid retrieval.  

- **Grade Documents Node (`grade_documents_node.py`)**  
  Evaluates the relevance of retrieved documents and filters out irrelevant/noisy chunks.  

- **Grade Answer Chain (`grade_answer_chain.py`)**  
  Verifies whether the generated answer is grounded in the retrieved evidence.  

- **Grade Hallucinations Chain (`grade_hallucinations_chain.py`)**  
  Detects hallucinations by comparing LLM outputs with retrieved source documents.  

- **Graph State (`graph_state2.py`)**  
  Maintains conversation state, retrieved docs, graded docs, and answers.  

- **Graph Orchestration (`graph_2.py`)**  
  Connects retriever, grader, generator into a pipeline ensuring correction before output.  

#### Workflow

1. **User Query → Retriever** → fetch candidate docs.  
2. **Document Grader** → filter documents with low relevance scores.  
3. **Answer Generator (`generate_node2.py`)** → produce candidate answer.  
4. **Answer Grader** → check if the answer is consistent with docs.  
5. If poor → **re-trigger retrieval** or **web search fallback**.  
6. Final **corrected answer** returned.  

---

### 🔄 Adaptive RAG

Adaptive RAG dynamically **chooses the best strategy** (retrieval, generation, correction, or external search) depending on query type and quality of results.  

#### Components

- **Transform Query Node (`transform_query_node.py`)**  
  Rewrites user queries for better retrieval.  

- **Query Route Chain (`query_route_chain.py`)**  
  Decides if query should be handled by **retriever** or **web search**.  

- **Web Search Node (`web_search_node.py`)**  
  Uses external search (Tavily API or other engines) for out-of-domain queries.  

- **Grader Chain (`grader_chain.py`)**  
  Unified grading for documents, answers, and hallucination detection.  

- **Generate Node (`generate_node2.py`)**  
  Produces final answer from best available context.  

- **Graph Orchestration (`graph_2.py`)**  
  Implements adaptive logic:  
  - If retrieval high quality → answer directly.  
  - If retrieval weak → re-query or escalate to web search.  
  - If hallucination risk high → self-correct before finalizing.  

#### Workflow

1. **User Query** → optional **query rewriting**.  
2. **Route Decision** → Milvus retrieval vs. external web search.  
3. **Grading Layer** → check doc and answer quality.  
4. **Adaptive Control** → choose whether to:  
   - Use retrieved docs  
   - Retry with rewritten query  
   - Fall back to web search  
5. **Answer Generation** → produce grounded, corrected, adaptive response.  

---

✅ With **Corrective RAG**, the system ensures **answer quality**.  
✅ With **Adaptive RAG**, the system ensures **flexibility and robustness** across domains.  








