# frontend/app.py
"""
Streamlit frontend for the 3GPP Chatbot RAG.
Connects to FastAPI backend at http://localhost:8000/chat.
"""

import streamlit as st
import requests

st.title("3GPP Chatbot RAG")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
pergunta = st.chat_input("Ask about 3GPP specifications...")

if pergunta:
    # Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://localhost:5000/chat",
                    json={"message": pergunta}
                )
                response.raise_for_status()
                data = response.json()

                resposta = data.get("response", "No response received")
                fontes = data.get("sources", [])

                # Display main response
                st.markdown(resposta)

                # Display sources if available
                if fontes:
                    st.markdown("**Sources used:**")
                    for fonte in fontes:
                        st.markdown(
                            f"- **Spec:** {fonte.get('spec', 'n/a')} | "
                            f"**Release:** {fonte.get('release', 'n/a')} | "
                            f"**Series:** {fonte.get('series', 'n/a')} | "
                            f"**Chunk Index:** {fonte.get('chunk_index', 'n/a')}"
                        )

                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": resposta})

            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to backend: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")