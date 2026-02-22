# src/app/graph/rag_graph.py
"""
LangGraph RAG pipeline for 3GPP specifications chatbot.
Uses get_relevant_documents (already formatted as strings) for retrieval.
"""
from typing import Annotated, List, Dict, TypedDict, Generator, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from app.retrieval.retriever import get_relevant_documents
from app.config.settings import OPENAI_API_KEY, logger


class RAGState(TypedDict):
    question: str
    context: str  # now string (joined formatted docs)
    answer: Generator[str, None, None]
    messages: Annotated[List[Dict], add_messages]


def retrieve_node(state: RAGState) -> Dict[str, Any]:
    """
    Retrieves formatted strings using get_relevant_documents.
    """
    logger.info(f"Graph - Retrieving for: {state['question']}")
    formatted_strings = get_relevant_documents(
        query=state["question"],
        k=5
    )
    context = "\n\n---\n\n".join(formatted_strings)
    logger.info(f"Graph - Retrieved and formatted {len(formatted_strings)} chunks")
    return {"context": context}


def generate_node(state: RAGState) -> Dict[str, Any]:
    """
    Generates streaming response using pre-formatted context.
    """
    logger.info("Graph - Starting generation phase...")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in 3GPP technical specifications. "
                   "Answer the question based only on the provided context. "
                   "Be precise, technical, and concise. "
                   "At the end, list sources in format: "
                   "(Spec: <spec>, Release: <release>, Series: <series>, Chunk Index: <chunk_index>)."),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
    ])

    chain = prompt | llm | StrOutputParser()

    answer_stream = chain.stream({
        "question": state["question"],
        "context": state["context"]
    })

    logger.info("Graph - Generation started (streaming tokens)...")

    return {"answer": answer_stream}


def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


COMPILED_GRAPH = build_rag_graph()


def run_rag(question: str) -> Generator[Dict[str, Any], None, None]:
    logger.info(f"RAG execution started for question: {question}")

    initial_state: RAGState = {"question": question, "messages": []}

    for event in COMPILED_GRAPH.stream(initial_state):
        if "generate" in event:
            answer_stream = event["generate"]["answer"]
            for token in answer_stream:
                yield {"type": "token", "data": token}

        if END in event:
            # Final sources (from metadata - but since formatted in strings, we can't extract easily)
            # If you still want sources, can return them separately or log them
            logger.info("RAG execution completed.")
            yield {"type": "end", "data": None}