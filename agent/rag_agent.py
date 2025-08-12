from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from My_RAG_Project.documents.milvus_db_pdf import MilvusPDFWriter
from My_RAG_Project.utils.env_utils import COLLECTION_NAME
from My_RAG_Project.utils.log_utils import log


# -----------------------------
# 1) Prepare retriver
# -----------------------------
def _build_pdf_retriever(k: int = 5, expr: str = "page_number >= 1"):
    mv = MilvusPDFWriter()
    mv.create_connection()

    retriever = mv.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "expr": expr,
        },
    )
    return retriever


def pdf_retrieve_fn(query: str, k: int = 5, expr: str = "page_number >= 1") -> str:
    retriever = _build_pdf_retriever(k=k, expr=expr)

    try:
        docs = retriever.get_relevant_documents(query)
    except Exception as e:
        log.error(f"PDF retriever failed: {e}")
        return f"[检索失败] {e}"

    lines: List[str] = []
    for i, d in enumerate(docs):
        page = d.metadata.get("page_number", "?")
        kw = d.metadata.get("keywords", "")
        preview = d.page_content[:300].replace("\n", " ")
        lines.append(f"[{i}] page={page} | kw={kw} | {preview}...")
    pretty = "\n".join(lines) if lines else "[无检索结果]"
    return pretty


# Encapsulate the retrieval function as a LangChain Tool for Agent to call
pdf_retriever_tool: BaseTool = Tool(
    name="pdf_retriever",
    func=lambda q: pdf_retrieve_fn(q, k=5, expr="page_number >= 1"),
    description=(
        "面向 PDF 知识库的检索工具。输入自然语言问题，返回与问题最相关的文档片段，"
        "包含页码、关键词和片段摘要。适合回答基于上传 PDF 的问题。"
    ),
)


# -----------------------------
# 2) Prompt template
# -----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个智能助手，尽可能地调用工具回答用户的问题。必要时引用检索到的证据。"),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
    ]
)


# -----------------------------
# 3) Build Agent
# -----------------------------
def build_pdf_agent() -> AgentExecutor:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    agent = create_tool_calling_agent(
        llm=llm,
        tools=[pdf_retriever_tool],
        prompt=prompt,
    )
    executor = AgentExecutor(agent=agent, tools=[pdf_retriever_tool])
    return executor


# -----------------------------
# 4) chat history
# -----------------------------
_store: Dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


def build_agent_with_history() -> RunnableWithMessageHistory:
    executor = build_pdf_agent()
    runnable = RunnableWithMessageHistory(
        runnable=executor,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return runnable



if __name__ == "__main__":
    log.info(f"Using collection: {COLLECTION_NAME}")

    agent_with_history = build_agent_with_history()

    # First Round
    res = agent_with_history.invoke(
        {"input": "万达智慧商业平台的核心功能有哪些？"},
        config={"configurable": {"session_id": "demo_user_1"}},
    )
    print("\n=== Agent ===")
    print(res["output"])

    # Second Round, use chat history
    res2 = agent_with_history.invoke(
        {"input": "上面提到的“直播互动”具体是怎么做的？"},
        config={"configurable": {"session_id": "demo_user_1"}},
    )
    print("\n=== Agent (follow-up) ===")
    print(res2["output"])
