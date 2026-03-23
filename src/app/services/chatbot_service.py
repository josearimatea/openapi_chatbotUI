# src/app/services/chatbot_service.py
"""
Chatbot service — executes the compiled LangGraph pipeline and delivers results.

This is the bridge between the API layer (routes.py) and the graph (chatbot_graph.py).
The API calls chat(), which runs the graph and yields events one at a time.

Call chain:
    api/routes.py  →  chat(question)  →  COMPILED_GRAPH.stream()
                                              │
                                    orchestrator → retrieve → answer

Output format:
    chat() is a generator that yields dicts:
        {"type": "token", "data": "The"}      ← one piece of the response text
        {"type": "token", "data": " input"}
        {"type": "token", "data": " for"}
        ...
        {"type": "end",   "data": None}        ← signals completion

    routes.py then either:
        - /chat:        collects all tokens into a single string → returns JSON
        - /chat/stream: forwards each dict as SSE → frontend shows tokens in real-time
"""

from typing import Dict, Any, Generator

from app.config import get_logger
from app.graph.chatbot_graph import COMPILED_GRAPH

logger = get_logger(__name__)


def chat(question: str) -> Generator[Dict[str, Any], None, None]:
    """
    Executes the chatbot RAG pipeline for a given question.

    Iterates over COMPILED_GRAPH.stream() events and yields standardized
    {"type": ..., "data": ...} dicts for the API layer.

    Two possible paths through the graph:
        1. Casual:    orchestrator responds directly → tokens from casual_response
        2. Technical: orchestrator → retrieve → answer → tokens from LLM stream
    """
    logger.info(f"Chat started for question: {question}")
    initial_state = {"question": question, "messages": []}

    for event in COMPILED_GRAPH.stream(initial_state):

        # Technical path: the answer node produced a streaming generator
        # event looks like: {"answer": {"answer": <generator of str tokens>}}
        if "answer" in event:
            answer_stream = event["answer"]["answer"]
            for token in answer_stream:
                yield {"type": "token", "data": token}

        # Casual path: orchestrator decided no retrieval needed
        # event looks like: {"orchestrator": {"needs_retrieval": False, "casual_response": "Hi!"}}
        elif "orchestrator" in event:
            casual_response = event["orchestrator"].get("casual_response", "")
            if casual_response and not event["orchestrator"].get("needs_retrieval", True):
                # Split into words to simulate token-by-token delivery
                for word in casual_response.split():
                    yield {"type": "token", "data": word + " "}

    logger.info("Chat completed.")
    yield {"type": "end", "data": None}
