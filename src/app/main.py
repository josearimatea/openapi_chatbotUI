# src/app/main.py
"""
FastAPI application factory.
Creates the app, configures middleware, and includes routers.

This is the entry point for uvicorn:
    uvicorn app.main:app --reload --port 5000

Full request flow:
    User (Streamlit) → main.py (FastAPI app) → api/routes.py → services/chatbot_service.py
                                                                    → graph/chatbot_graph.py
                                                                        → graph/nodes.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_logger, APP_ENV, log_level

logger = get_logger(__name__)

logger.debug("Debug message → se voce vir isso, LOG_LEVEL esta realmente em DEBUG")
logger.info(f"APP_ENV={APP_ENV} | LOG_LEVEL={log_level}")

app = FastAPI(title="3GPP RAG Chatbot API")

# Enable CORS for frontend access (Streamlit, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routes from api/routes.py (e.g., /chat, /chat/stream)
app.include_router(router)
