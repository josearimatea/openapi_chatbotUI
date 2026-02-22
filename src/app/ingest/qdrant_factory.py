"""
Factory class for embeddings and QdrantVectorStore.
Centralizes creation of embeddings and vector store to avoid duplication.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from app.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME

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
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    def get_qdrant_vector_store(self, collection_name: str = COLLECTION_NAME) -> QdrantVectorStore:
        """
        Returns ready QdrantVectorStore with named vectors.
            - vector_name: name for dense vectors (default "text-dense")
            - sparse_vector_name: name for sparse vectors (default "text-sparse")
            This allows consistent naming across the app and easy updates if needed.
        """
        return QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings,
            vector_name="text-dense",
            sparse_vector_name="text-sparse",
        )