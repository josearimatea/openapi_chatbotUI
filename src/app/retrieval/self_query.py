# Only used with SelfQueryRetriever - not used for direct Qdrant retrieval using vector_store.similarity_search

from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="release",
        description="3GPP specification release (e.g., 'Rel-18')",
        type="string",
    ),
    AttributeInfo(
        name="series",
        description="Specification series (e.g., '28_series')",
        type="string",
    ),
    AttributeInfo(
        name="spec",
        description="Specification name (e.g., '28532')",
        type="string",
    ),
]

document_content_description = "Processed content of 3GPP specifications"