# main.py
"""
Main FastAPI application for the 3GPP Chatbot RAG API.
Provides /chat endpoint using the LangGraph RAG pipeline.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.graph.rag_graph import run_rag
from app.utils.settings import logger

app = FastAPI(title="3GPP RAG Chatbot API")

# Enable CORS for frontend access (Streamlit, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production (e.g., ["http://localhost:8501"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    message: str


@app.get("/")
def home():
    """Simple health check endpoint."""
    return {"message": "3GPP RAG Chatbot API is running! Use POST /chat"}


@app.post("/chat")
def chat(question: Question):
    """
    Main chat endpoint.
    Processes the question using the RAG pipeline (LangGraph) and returns the response.
    """
    user_query = question.message.strip()

    if not user_query:
        logger.warning("Empty question received")
        return {"response": "Empty question."}

    logger.info(f"Received question: {user_query}")

    try:
        full_response = ""
        sources = []

        # Run the RAG pipeline
        for event in run_rag(user_query):
            if event["type"] == "token":
                full_response += event["data"]
            elif event["type"] == "sources":
                sources = event["data"]

        logger.info(f"RAG completed. Response length: {len(full_response)} chars")

        return {
            "response": full_response.strip(),
            "sources": sources
        }

    except Exception as e:
        logger.error(f"RAG pipeline failed: {str(e)}", exc_info=True)
        return {"response": "Sorry, an error occurred while processing your question."}