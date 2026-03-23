# src/app/infrastructure/qdrant.py
"""
Factory class for embeddings and QdrantVectorStore.
Centralizes creation of embeddings and vector store to avoid duplication.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME
from app.config.settings import EMBEDDING_MODEL


class QdrantFactory:
    def __init__(self, device: str = "cpu"):
        """
        Initialize with device for embeddings (cpu or cuda).
        Creates Qdrant client and embeddings model.
        """
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            timeout=120,
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    def get_qdrant_vector_store(self, collection_name: str = COLLECTION_NAME) -> QdrantVectorStore:
        """
        Returns ready QdrantVectorStore with named vectors.
        """
        return QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings,
            vector_name="text-dense",
            sparse_vector_name="text-sparse",
        )


@lru_cache(maxsize=1)
def get_factory(device: str = "cpu") -> QdrantFactory:
    """
    Returns a cached QdrantFactory instance (singleton pattern).

    lru_cache memorizes the result of the first call. On subsequent calls
    with the same device argument, it returns the cached instance instead
    of creating a new QdrantClient + loading the embeddings model again.
    This avoids ~3-5s of overhead per request.
    """
    return QdrantFactory(device=device)
