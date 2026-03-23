# src/app/graph/chatbot_graph.py
"""
LangGraph RAG pipeline definition.
Defines the graph structure (nodes + edges) and compiles it.

Graph flow:
    ┌──────────────┐
    │ orchestrator  │  Agent 1: classifies query (casual or technical)
    └──────┬───────┘
           │
           ├── needs_retrieval = false
           │   → END (casual_response already in state)
           │
           └── needs_retrieval = true
               │
        ┌──────┴───────┐
        │   retrieve    │  Vector search in Qdrant (not an LLM agent)
        └──────┬───────┘
               │
        ┌──────┴───────┐
        │    answer     │  Agent 2: generates RAG response from retrieved docs
        └──────┬───────┘
               │
              END

The compiled graph (COMPILED_GRAPH) is imported by services/chatbot_service.py
which executes it via COMPILED_GRAPH.stream().
"""

from langgraph.graph import StateGraph, END

from app.graph.state import RAGState
from app.graph.nodes import (
    orchestrator_node,
    retrieve_node,
    answer_node,
    route_after_orchestrator,
)


def build_chatbot_graph():
    """Builds and compiles the RAG chatbot graph."""
    graph = StateGraph(RAGState)

    # Register the 3 nodes
    graph.add_node("orchestrator", orchestrator_node)   # Agent 1: classify
    graph.add_node("retrieve", retrieve_node)           # Vector search (no LLM)
    graph.add_node("answer", answer_node)               # Agent 2: generate answer

    # Entry point: every question starts at the orchestrator
    graph.set_entry_point("orchestrator")

    # Conditional edge: orchestrator decides the next step
    # - "retrieve" if technical (needs documents from Qdrant)
    # - "__end__"  if casual   (orchestrator already has the response)
    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "retrieve": "retrieve",
            "__end__": END,
        }
    )

    # Linear edges: retrieve always goes to answer, answer always ends
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


# Compiled once at import time — reused for every request
COMPILED_GRAPH = build_chatbot_graph()
