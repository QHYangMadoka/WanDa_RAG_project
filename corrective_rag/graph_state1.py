from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class AgentState(TypedDict):
    """
    Global state for the corrective RAG graph.

    - `messages` accumulates all dialogue/tool messages across the workflow.
      `add_messages` means new messages will be appended.
    """
    messages: Annotated[List[BaseMessage], add_messages]


class Grade(BaseModel):
    """
    Binary relevance grade for retrieved context.
    """
    binary_score: str = Field(description="Relevance judgement: 'yes' or 'no'")