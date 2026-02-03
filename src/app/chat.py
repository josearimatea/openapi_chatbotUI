# app/chat.py
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_gpt(question: str, model: str = "gpt-4o-mini") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an assistant that answers questions about 3GPP"},
            {"role": "user", "content": question}
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content