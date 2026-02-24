# For Reproduction

## Development Setup

1. Clone the repository:
$ git clone https://github.com/seu-usuario/openapi_chatbotUI.git
$ cd openapi_chatbotUI
   
2. Create and sync the environment:
$ uv sync

3. Install the project in editable mode (so imports work):
$ uv pip install --editable .

## Run Qdrant

1 - You need to run the Qdrant Docker container in order to use embeddings:

$ docker pull qdrant/qdrant

$ docker run -d --name qdrant-local \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant

2 - You must recreate the embeddings on the new machine or server that will run the system.

# For Usage

1 - Running Backend and Frontend:

i) First, run Qdrant:

$ docker start qdrant-local

ii) Then run the Backend and Frontend setup:

Terminal 1 – FastAPI Backend
$ uv run uvicorn app.main:app --reload --port 5000

Terminal 2 – Streamlit Frontend
$ uv run streamlit run frontend/app.py --server.port 5001

Or use the script that starts both:

uv run python scripts/start_app.py
