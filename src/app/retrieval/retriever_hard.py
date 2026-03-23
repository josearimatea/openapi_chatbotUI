"""
Retriever module for semantic search from Qdrant.
- retrieve: core logic (manual filter or deterministic extraction + manual search)
- get_relevant_documents: reuses retrieve, formats each doc as string with metadata header,
  and returns List[str] ready for prompt/context (used by LangGraph)

This retriever hard drops SelfQueryRetriever in favor of a simpler, deterministic approach
However, the principal and best script still remains retriever.py, which is used by the graph.
This retriever_hard.py is a simplified version for testing and comparison.
"""

from typing import Dict, List, Optional, Any
from langchain_core.documents import Document
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.config import get_logger
from app.infrastructure.llm import llm
from app.infrastructure.hardware import device
from app.infrastructure.qdrant import QdrantFactory
from app.retrieval.self_query import metadata_field_info, document_content_description  # Kept for future use

logger = get_logger(__name__)


def extract_filters_from_query(query: str) -> Optional[Dict[str, str]]:
    """
    Extracts potential metadata filters deterministically using a dedicated LLM chain.
    Returns dict or None if no filters were found.
    """
    parser = JsonOutputParser()

    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an extractor for 3GPP query filters. "
                   "Extract ONLY if clearly mentioned in the query: release, series, spec. "
                   "Output JSON with null for missing fields. "
                   "{format_instructions}\n"
                   "Examples:\n"
                   "- Query: 'createMOI in release 18' -> {{\"release\": \"Rel-18\", \"series\": null, \"spec\": null}}\n"
                   "- Query: 'TS 28.532 handover in Rel-17' -> {{\"release\": \"Rel-17\", \"series\": \"28_series\", \"spec\": \"28532\"}}\n"
                   "- Query: 'general 5G info' -> {{\"release\": null, \"series\": null, \"spec\": null}}"),
        ("human", "Query: {query}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = extract_prompt | llm | parser
    result = chain.invoke({"query": query})

    filters = {k: v for k, v in result.items() if v is not None}
    if not filters:
        return None

    logger.info(f"Extracted filters: {filters}")
    return filters


def format_doc_as_string(doc: Document) -> str:
    """
    Formats a single document as a string with metadata header + content.
    Returns plain string (not Document object).
    """
    md = doc.metadata or {}
    header = (
        f"release: {md.get('release', 'unknown')}\n"
        f"series: {md.get('series', 'unknown')}\n"
        f"spec: {md.get('spec', 'unknown')}\n"
        f"chunk_index: {md.get('chunk_index', 'unknown')}\n"
        f"\n"
    )
    return header + doc.page_content.strip()


def retrieve(
    query: str,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Core retrieval logic.
    - If filters provided: manual filter mode
    - Else: Deterministic extraction of filters + manual search (no SelfQueryRetriever)
    Returns structured dict with raw docs, generated_query, generated_filter (same signature as your original code).
    """
    logger.info(f"Starting retrieval for query: {query} | k={k} | filters={filters}")

    factory = QdrantFactory(device=device)
    vector_store = factory.get_qdrant_vector_store()

    # Use extracted filters if no manual filters were provided
    if not filters:
        filters = extract_filters_from_query(query) or {}

    # Always use manual mode now (deterministic and simple)
    if filters:
        logger.info("Using manual filter mode with provided/extracted filters")
        qdrant_filter = Filter(
            must=[
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filters.items()
            ]
        )
    else:
        logger.info("No filters - Using pure similarity search")
        qdrant_filter = None

    docs = vector_store.similarity_search(query=query, k=k, filter=qdrant_filter)

    result = {
        "docs": docs,
        "generated_query": query,  # No rewrite, as we're dropping SelfQuery
        "generated_filter": f"Filters: {filters}" if filters else "No filter applied",
    }

    logger.info(f"Retrieval finished. Found {len(result['docs'])} documents.")
    logger.info(f"Generated query: {result['generated_query']}")
    logger.info(f"Generated filter: {result['generated_filter']}")

    return result


def get_relevant_documents(
    query: str,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    LangGraph integration point.
    Calls retrieve, formats each document as string with metadata header,
    and returns List[str] ready for prompt/context.
    """
    result = retrieve(query=query, k=k, filters=filters)

    # Format each document as string (header + content)
    formatted_strings = [format_doc_as_string(doc) for doc in result["docs"]]

    logger.info(f"get_relevant_documents: formatted {len(formatted_strings)} strings for prompt")

    return formatted_strings