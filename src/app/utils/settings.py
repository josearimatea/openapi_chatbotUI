# src/app/utils/settings.py

import os
import logging
import torch
from langchain_openai import ChatOpenAI

# Existing settings
DATA_DIRECTORY = '../../../../../Dataset/TSpec-LLM/3GPP-clean'
CHUNKS_FILE = '../../../files/tspec_chunks.pkl'
QDRANT_HOST = 'localhost'
QDRANT_PORT = 6333
COLLECTION_NAME = '3gpp_rel18_28'

# Load OpenAI API key from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# LLM instance (global, reusable)
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=OPENAI_API_KEY
)

# Device for embeddings (GPU if available)
torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using Device: {device}")

# Logging configuration (added here - central and reusable)
logging.basicConfig(
    level=logging.INFO,                    
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler()            # Console output
        # logging.FileHandler("rag_app.log") 
    ]
)

# Logger global (use logger.info(), logger.debug(), etc.)
logger = logging.getLogger("rag_project")
logger.setLevel(logging.INFO)