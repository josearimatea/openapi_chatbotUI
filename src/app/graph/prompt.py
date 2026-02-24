from langchain.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template(
    "Answer the question based on the context: {context}\nQuestion: {question}"
)