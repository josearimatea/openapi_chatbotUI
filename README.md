# Para reprodução

## Setup de Desenvolvimento

1. Clone o repositório:
$ git clone https://github.com/seu-usuario/openapi_chatbotUI.git
$ cd openapi_chatbotUI
   
2. Crie e sincronize o ambiente:
$ uv sync

3. Instale o projeto em modo editável (para imports funcionarem):
$ uv pip install --editable .

## Rodar Qdrant

1 - Necessário rodar o docker do Qadrant para poder usar Embeddings:

$ docker pull qdrant/qdrant

$ docker run -d --name qdrant-local \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant

2 - Precisa criar os embeddings novamente na nova máquina ou servidor que irá rodar o sistema.

# Para uso

1 - Rodando Backend e Frontend:

Rode o Qdrant:

$ docker run -d --name qdrant-local \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant

$ docker start qdrant-local


Terminal 1 – Backend FastAPI
$ uv run uvicorn app.main:app --reload --port 5000

Terminal 2 – Frontend Streamlit
$ uv run streamlit run frontend/app.py --server.port 5001


