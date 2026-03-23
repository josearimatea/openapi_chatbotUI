# src/app/api/routes.py
"""
API route definitions (HTTP layer).
Receives requests, calls chatbot_service, and returns responses.

Endpoints:
    GET  /             → health check
    POST /chat         → non-streaming (returns full JSON response)
    POST /chat/stream  → streaming via SSE (sends tokens as they are generated)

Data flow:
    Frontend (Streamlit)
        → POST /chat or /chat/stream with {"message": "user question"}
            → chatbot_service.chat(question)
                → yields {"type": "token", "data": "word"} events
            → /chat collects all tokens into a single string and returns JSON
            → /chat/stream forwards each event as SSE to the frontend in real-time
"""

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.schemas import Question
from app.services.chatbot_service import chat as chatbot_chat, clear_memory
from app.config import get_logger, MODEL, EMBEDDING_MODEL, COLLECTION_NAME

logger = get_logger(__name__)

router = APIRouter()


@router.get("/")
def home():
    """Simple health check endpoint."""
    return {"message": "3GPP RAG Chatbot API is running! Use POST /chat"}


@router.get("/info")
def info():
    """
    Returns current model and infrastructure settings.
    Used by the frontend sidebar to display dynamic info instead of hardcoded strings.
    """
    return {
        "model": MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "collection_name": COLLECTION_NAME,
        "orchestration": "LangGraph",
        "vector_db": "Qdrant",
    }


@router.post("/clear")
def clear(thread_id: str = "default"):
    """
    Clears conversation memory for a given thread_id.
    Called by the frontend when the user clicks "Clear conversation".
    """
    clear_memory(thread_id)
    logger.info(f"Memory cleared for thread: {thread_id}")
    return {"status": "ok"}


@router.post("/chat")
def chat(question: Question):
    """
    Non-streaming chat endpoint.
    Collects all tokens from the service and returns a complete JSON response.
    Used by clients that don't support streaming.
    """
    user_query = question.message.strip()

    if not user_query:
        logger.warning("Empty question received")
        return {"response": "Empty question."}

    logger.info(f"Received question: {user_query}")

    try:
        full_response = ""
        sources = []

        # chatbot_chat() yields dicts like {"type": "token", "data": "word"}
        # We collect all tokens into a single string
        for event in chatbot_chat(user_query, thread_id=question.thread_id):
            if event["type"] == "token":
                full_response += event["data"]
            elif event["type"] == "sources":
                sources = event["data"]

        logger.info(f"RAG completed. Response length: {len(full_response)} chars")

        # Return plain JSON — the user never sees the internal dict format
        return {
            "response": full_response.strip(),
            "sources": sources
        }

    except Exception as e:
        logger.error(f"RAG pipeline failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing your question.")


@router.post("/chat/stream")
def chat_stream(question: Question):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    Each token is sent to the frontend as it is generated, enabling
    real-time display (like ChatGPT typing effect).

    SSE format: each line is "data: {json}\n\n"
    The frontend reads these lines with requests.post(stream=True).
    """
    user_query = question.message.strip()

    if not user_query:
        logger.warning("Empty question received (stream)")
        return StreamingResponse(
            iter([f"data: {json.dumps({'type': 'error', 'data': 'Empty question.'})}\n\n"]),
            media_type="text/event-stream",
        )

    logger.info(f"Received question (stream): {user_query}")

    def event_generator():
        """Wraps chatbot_chat() events in SSE format for the frontend."""
        try:
            for event in chatbot_chat(user_query, thread_id=question.thread_id):
                # Each event becomes an SSE line: "data: {"type":"token","data":"word"}\n\n"
                yield f"data: {json.dumps(event)}\n\n"
            yield f"data: {json.dumps({'type': 'end', 'data': None})}\n\n"
        except Exception as e:
            logger.error(f"RAG pipeline failed (stream): {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",       # Don't cache the stream
            "Connection": "keep-alive",         # Keep TCP connection open
            "X-Accel-Buffering": "no",          # Disable proxy buffering (nginx, etc.)
        },
    )
