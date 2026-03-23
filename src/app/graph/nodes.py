# src/app/graph/nodes.py
"""
LangGraph node functions for the RAG pipeline.
Each node reads from the shared state (RAGState) and writes its output fields.

Pipeline flow (3 nodes, 2 LLM agents max):

    [orchestrator]  ← Agent 1 (LLM): classifies query as casual or technical
         │
         ├── casual → END (response already in state.casual_response)
         │
         └── technical
              │
         [retrieve]  ← Vector search in Qdrant (no LLM call)
              │
         [answer]    ← Agent 2 (LLM): generates response from retrieved docs
              │
             END

Node inputs/outputs (what each node reads and writes in RAGState):

    orchestrator:  reads  → question
                   writes → needs_retrieval, casual_response

    retrieve:      reads  → question
                   writes → context

    answer:        reads  → question, context
                   writes → answer (streaming token generator)

Graph definition: graph/chatbot_graph.py
Execution:        services/chatbot_service.py
"""

from typing import Dict, Any

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from app.config import get_logger, NUMBER_RETRIEVE_CHUNKS
from app.infrastructure.llm import llm
from app.retrieval.retriever import get_relevant_documents
from app.prompts.chatbot_prompts import ORCHESTRATOR_PROMPT, ANSWER_PROMPT

logger = get_logger(__name__)


def orchestrator_node(state: dict) -> Dict[str, Any]:
    """
    Agent 1 — Classifies the query and responds directly if casual.

    Uses ORCHESTRATOR_PROMPT which asks the LLM to return JSON:
        {"needs_retrieval": true/false, "response": "..."}

    If casual: needs_retrieval=False, response="Oi! Como posso ajudar?"
    If technical: needs_retrieval=True, response="" (empty)
    """
    logger.info(f"Graph - Orchestrating for: {state['question']}")

    parser = JsonOutputParser()
    chain = ORCHESTRATOR_PROMPT | llm | parser
    result = chain.invoke({"question": state["question"]})

    needs_retrieval = result.get("needs_retrieval", True)
    casual_response = result.get("response", "")

    logger.info(f"Graph - Needs retrieval: {needs_retrieval}")
    return {"needs_retrieval": needs_retrieval, "casual_response": casual_response}


def retrieve_node(state: dict) -> Dict[str, Any]:
    """
    Vector search — Retrieves relevant 3GPP documents from Qdrant.

    NOT an LLM agent. Uses SelfQueryRetriever for semantic search
    with automatic metadata filtering (release, series, spec).

    Returns k documents (NUMBER_RETRIEVE_CHUNKS) formatted as strings
    with metadata headers, joined by "---" separators.
    """
    logger.info(f"Graph - Retrieving for: {state['question']}")
    formatted_strings = get_relevant_documents(
        query=state["question"],
        k=NUMBER_RETRIEVE_CHUNKS
    )
    context = "\n\n---\n\n".join(formatted_strings)
    logger.info(f"Graph - Retrieved and formatted {len(formatted_strings)} chunks")
    return {"context": context}


def answer_node(state: dict) -> Dict[str, Any]:
    """
    Agent 2 — Generates the final RAG response from retrieved documents.

    Uses ANSWER_PROMPT (different from orchestrator's prompt) which is
    focused on technical accuracy and source citation.

    Returns a streaming generator of string tokens — each token is a
    small piece of text (e.g., "The", " input", " for") that the service
    delivers to the frontend one at a time.
    """
    logger.info("Graph - Starting answer generation...")
    chain = ANSWER_PROMPT | llm | StrOutputParser()
    answer_stream = chain.stream({
        "question": state["question"],
        "context": state["context"]
    })
    logger.info("Graph - Answer generation started (streaming tokens)...")
    return {"answer": answer_stream}


def route_after_orchestrator(state: dict) -> str:
    """
    Conditional routing — called by LangGraph after orchestrator_node.

    Decides the next step based on the orchestrator's classification:
        - needs_retrieval=True  → "retrieve" (technical path)
        - needs_retrieval=False → "__end__"  (casual path, skip to END)
    """
    return "retrieve" if state["needs_retrieval"] else "__end__"
