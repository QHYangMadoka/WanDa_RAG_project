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

#### 🐳 Step-by-step Installation

Open your Ubuntu WSL terminal and run the following commands:

```bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

















