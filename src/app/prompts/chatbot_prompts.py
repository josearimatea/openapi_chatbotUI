# src/app/prompts/chatbot_prompts.py
"""
Prompt templates for the RAG pipeline.
Centralized here for easy iteration and testing.

Each prompt is used by a specific node in graph/nodes.py:
    ORCHESTRATOR_PROMPT → orchestrator_node (Agent 1: classification)
    ANSWER_PROMPT       → answer_node       (Agent 2: RAG response)
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# ── Orchestrator prompt ──────────────────────────────────────
# Agent 1: Classifies the query as casual or technical.
#
# Expected LLM output (JSON):
#   {"needs_retrieval": true/false, "response": "..."}
#
# - Casual query  → needs_retrieval=false, response="Oi! Como posso ajudar?"
# - Technical     → needs_retrieval=true,  response="" (empty)

_orchestrator_parser = JsonOutputParser()

ORCHESTRATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly 3GPP specs chatbot assistant. "
               "If the query is casual, simple, or non-technical (e.g., greetings, general chat), "
               "provide a helpful response and set needs_retrieval to false. "
               "If it requires technical information from 3GPP specifications, "
               "do not answer, leave response empty, and set needs_retrieval to true. "
               "{format_instructions}\n"
               "Examples:\n"
               "- Query: 'Oi' -> {{\"response\": \"Oi! Como posso ajudar hoje?\", \"needs_retrieval\": false}}\n"
               "- Query: 'What is the input for createMOI?' -> {{\"response\": \"\", \"needs_retrieval\": true}}\n"
               "- Query: 'How are you?' -> {{\"response\": \"I'm good, thanks! Ready to help with 3GPP queries.\", \"needs_retrieval\": false}}\n"
               "- Query: 'Explain 5G beamforming' -> {{\"response\": \"\", \"needs_retrieval\": true}}"),
    ("human", "Query: {question}"),
]).partial(format_instructions=_orchestrator_parser.get_format_instructions())


# ── Answer prompt ────────────────────────────────────────────
# Agent 2: Generates the final RAG answer from retrieved documents.
#
# Input variables:
#   {question} → the user's original question
#   {context}  → retrieved documents from Qdrant (formatted with metadata headers)
#
# Expected LLM output: free-form text with source citations at the end.

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in 3GPP technical specifications. "
               "Answer the question based only on the provided context. "
               "Be precise, technical, and concise. "
               "At the end, list sources in format: "
               "(Spec: <spec>, Release: <release>, Series: <series>, Chunk Index: <chunk_index>)."),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
])
