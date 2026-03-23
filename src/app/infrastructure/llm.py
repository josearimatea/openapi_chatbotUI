# src/app/infrastructure/llm.py
"""
LLM client initialization.
Creates and exposes the global LLM instance.
"""

from langchain_openai import ChatOpenAI

from app.config.settings import OPENAI_API_KEY, MODEL, TEMPERATURE

llm = ChatOpenAI(
    model=MODEL,
    temperature=TEMPERATURE,
    api_key=OPENAI_API_KEY
)
