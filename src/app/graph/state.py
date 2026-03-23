# src/app/graph/state.py
"""
LangGraph state definition for the RAG pipeline.

The state is a shared dictionary that flows between all nodes.
Each node reads what it needs and writes its output fields.

Fields and which node writes them:
    question         ← initial input (from user)
    needs_retrieval  ← orchestrator (True if technical, False if casual)
    casual_response  ← orchestrator (direct answer when casual, empty when technical)
    context          ← retrieve (formatted documents from Qdrant)
    answer           ← answer (streaming generator of response tokens)
    messages         ← LangGraph message history (managed automatically)
"""

from typing import Annotated, List, Dict, TypedDict, Generator
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    # Input: the user's question
    question: str

    # Orchestrator output: classification result
    needs_retrieval: bool                              # True = technical → retrieve + answer
    casual_response: str                               # Non-empty only when needs_retrieval=False

    # Retrieve output: documents from Qdrant joined as text
    context: str

    # Answer output: streaming token generator from the LLM
    answer: Generator[str, None, None]

    # LangGraph managed: conversation message history
    messages: Annotated[List[Dict], add_messages]
