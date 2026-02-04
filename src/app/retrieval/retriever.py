# retriever.py
"""
Reusable module for semantic search / retrieval from Qdrant.

Usage example:
    from retriever import get_relevant_chunks

    results = get_relevant_chunks("o que diz a especificação sobre beamforming?")
    for res in results:
        print(res["score"], res["text"][:200], "...")
"""

from typing import List, Dict, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from app.utils.settings import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME
from langchain_core.documents import Document



def retrieve(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict] = None,         # ex: {"release": "18"}
    score_threshold: float = 0.0,
) -> List[Dict]:
    """
    Retrieve the top-k most relevant chunks for a given query.

    Args:
        query: User question / search query
        top_k: Number of results to return (default: 5)
        filters: Optional metadata filters (e.g. {"release": "18", "series": "38"})
        score_threshold: Minimum similarity score to include (0.0 = all)

    Returns:
        List of dicts with: score, text, metadata (release, series, spec, etc.)
    """
    # 1. Initialize the same embeddings used during ingestion
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # ou "cuda"
        encode_kwargs={"normalize_embeddings": True},
    )

    # 2. Connect to Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)

    # 3. Create vector store instance with the correct named vector
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        vector_name="text-dense",
        sparse_vector_name="text-sparse",  # pode remover se não usar sparse
    )

    # 4. Prepare filter if provided
    qdrant_filter = None
    if filters:
        from qdrant_client.http.models import FieldCondition, MatchValue

        conditions = []
        for key, value in filters.items():
            conditions.append(
                FieldCondition(
                    key=f"payload.{key}",
                    match=MatchValue(value=value),
                )
            )
        if conditions:
            qdrant_filter = {"must": conditions}

    # 5. Retrieve documents
    docs = vector_store.similarity_search_with_score(
        query=query,
        k=top_k,
        filter=qdrant_filter,
    )

    # 6. Format results
    results = []
    for doc, score in docs:
        # score aqui é similaridade (quanto maior, melhor)
        # alguns preferem converter para distância: 1 - score
        results.append({
            "score": round(float(score), 4),
            "text": doc.page_content,
            "metadata": doc.metadata,
        })

    # Optional: filter by minimum score
    if score_threshold > 0:
        results = [r for r in results if r["score"] >= score_threshold]

    return results


def get_relevant_documents(query: str) -> List[Document]:
    """
    Função compatível com LangGraph.
    Chama retrieve internamente e converte para List[Document].
    """
    raw_results = retrieve(query=query, top_k=5)  # valores fixos por enquanto

    documents = []
    for res in raw_results:
        metadata = res["metadata"].copy()
        metadata["score"] = res["score"]

        doc = Document(
            page_content=res["text"],
            metadata=metadata
        )
        documents.append(doc)

    return documents