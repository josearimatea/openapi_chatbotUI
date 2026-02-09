DATA_DIRECTORY = '../../../../../Dataset/TSpec-LLM/3GPP-clean'
CHUNKS_FILE = '../../../files/tspec_chunks.pkl'
QDRANT_HOST = 'localhost'
QDRANT_PORT = 6333
COLLECTION_NAME = '3gpp_rel18_28'

import os
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

import torch
torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using Device: {device}")