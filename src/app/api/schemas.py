# src/app/api/schemas.py
"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel


class Question(BaseModel):
    message: str
