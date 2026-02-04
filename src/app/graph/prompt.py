from langchain.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template(
    "Responda à pergunta baseado no contexto: {context}\nPergunta: {question}"
)