# src/app/retrieval/retriever.py
"""
Retriever module for semantic search from Qdrant.

Main functions:
- retrieve:          core retrieval logic (SelfQuery or manual filter)
                     returns dict with docs, generated query and filter
- get_relevant_documents: prepares List[str] ready to be used in prompt / LangGraph
"""

from typing import Dict, List, Optional, Any

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.documents import Document
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from app.config import llm, device, get_logger
from app.ingest.qdrant_factory import QdrantFactory
from app.retrieval.self_query import metadata_field_info, document_content_description

logger = get_logger(__name__)


def build_self_query_retriever(k: int = 5) -> SelfQueryRetriever:
    """
    Builds SelfQueryRetriever using QdrantFactory.
    LLM parses natural language query → semantic search + metadata filters.
    """
    factory = QdrantFactory(device=device)
    vector_store = factory.get_qdrant_vector_store()

    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vector_store,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        enable_limit=True,
        search_kwargs={"k": k},
        verbose=True,           # shows parsed query / filter in console (useful for debug)
    )
    return retriever


def format_doc_as_string(doc: Document) -> str:
    """
    Formats a single Document as string with metadata header + content.
    Used to build context for LLM prompt.
    """
    md = doc.metadata or {}
    header = (
        f"release:     {md.get('release', 'unknown')}\n"
        f"series:      {md.get('series', 'unknown')}\n"
        f"spec:        {md.get('spec', 'unknown')}\n"
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
    Core retrieval function.

    Two modes:
    • filters provided   → manual exact filter + vector similarity
    • no filters         → SelfQueryRetriever (LLM decides filter + query)

    Returns dict with:
        "docs"             : List[Document]
        "generated_query"  : str
        "generated_filter" : str (human readable)
    """
    logger.info(f"Starting retrieval | query={query!r} | k={k} | filters={filters}")

    factory = QdrantFactory(device=device)
    vector_store = factory.get_qdrant_vector_store()

    if filters:
        logger.info("→ Using manual filter mode")
        qdrant_filter = Filter(
            must=[
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filters.items()
            ]
        )

        docs = vector_store.similarity_search(
            query=query,
            k=k,
            filter=qdrant_filter
        )

        result = {
            "docs": docs,
            "generated_query": query,
            "generated_filter": f"Manual filters: {filters}",
        }

    else:
        logger.info("→ Using SelfQueryRetriever (LLM parsing)")
        retriever = build_self_query_retriever(k=k)

        # Get the internal structured query the LLM generated
        structured_query = retriever.query_constructor.invoke({"query": query})

        docs = retriever.invoke(query)

        filter_str = str(structured_query.filter) if structured_query.filter else "No filter applied"

        result = {
            "docs": docs,
            "generated_query": structured_query.query or query,
            "generated_filter": filter_str,
        }

    logger.info(f"Retrieval finished → {len(result['docs'])} documents found")
    logger.debug(f"Generated query  : {result['generated_query']}")
    logger.debug(f"Generated filter : {result['generated_filter']}")

    return result


def get_relevant_documents(
    query: str,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Main entry point for LangGraph / prompt usage.

    Returns list of formatted strings (metadata header + content)
    ready to be joined and placed in the prompt.
    """
    result = retrieve(query=query, k=k, filters=filters)

    # IMPORTANT: result["docs"] is List[Document], not a single Document
    formatted_strings = [format_doc_as_string(doc) for doc in result["docs"]]

    logger.info(f"get_relevant_documents → formatted {len(formatted_strings)} strings")

    return formatted_strings