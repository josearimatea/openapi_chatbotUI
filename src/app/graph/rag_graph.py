from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.documents import Document
from app.graph.prompt import RAG_PROMPT
from app.retrieval.retriever import get_relevant_documents
from langchain_openai import ChatOpenAI
import os

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str

def retrieve(state):
    retriever = get_relevant_documents(ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    docs = retriever.get_relevant_documents(state["question"])
    return {"documents": docs}

def generate(state):
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chain = RAG_PROMPT | llm
    response = chain.invoke({"context": state["documents"], "question": state["question"]})
    return {"generation": response.content}

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()