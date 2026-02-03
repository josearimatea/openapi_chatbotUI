from fastapi import FastAPI
from pydantic import BaseModel
from .chat import ask_gpt
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Chatbot Simples - GPT")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Pergunta(BaseModel):
    mensagem: str

@app.get("/")
def home():
    return {"message": "Chatbot GPT rodando! Use POST /chat"}

@app.post("/chat")
def chat(pergunta: Pergunta):
    resposta = ask_gpt(pergunta.mensagem)
    return {"resposta": resposta}