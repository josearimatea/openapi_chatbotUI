# src/app/retrieval/retriever.py
"""
Retriever module for semantic search from Qdrant.
- retrieve: core logic (SelfQuery or manual, returns structured dict with raw docs, query, filter)
- get_relevant_documents: reuses retrieve, formats each doc as string with metadata header, returns List[str] for LangGraph
"""

from typing import Dict, List, Optional, Any
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.documents import Document
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from app.config.settings import llm, device, logger
from app.ingest.qdrant_factory import QdrantFactory
from app.retrieval.self_query import metadata_field_info, document_content_description


def build_self_query_retriever(k: int = 5) -> SelfQueryRetriever:
    """
    Builds SelfQueryRetriever using QdrantFactory.
    LLM parses query to generate semantic search + metadata filters automatically.
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
        verbose=True,  # shows parsed query/filter in console for debug
    )
    return retriever


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
    - Else: SelfQueryRetriever (LLM infers filters)
    Returns structured dict with raw docs, generated_query, generated_filter.
    """
    logger.info(f"Starting retrieval for query: {query} | k={k} | filters={filters}")

    factory = QdrantFactory(device=device)
    vector_store = factory.get_qdrant_vector_store()

    if filters:
        logger.info("Using manual filter mode")
        qdrant_filter = Filter(
            must=[
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filters.items()
            ]
        )

        docs = vector_store.similarity_search(query=query, k=k, filter=qdrant_filter)

        result = {
            "docs": docs,
            "generated_query": query,
            "generated_filter": f"Manual filters: {filters}",
        }
    else:
        logger.info("Using SelfQueryRetriever (LLM parsing)")
        retriever = build_self_query_retriever(k=k)

        structured_query = retriever.query_constructor.invoke({"query": query})

        docs = retriever.invoke(query)

        filter_str = str(structured_query.filter) if structured_query.filter else "No filter applied"

        result = {
            "docs": docs,
            "generated_query": structured_query.query or query,
            "generated_filter": filter_str,
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
    # formatted_strings = [format_doc_as_string(doc) for doc in result["docs"]]
    formatted_strings = format_doc_as_string(result["docs"])

    logger.info(f"get_relevant_documents: formatted {len(formatted_strings)} strings for prompt")

    return formatted_strings