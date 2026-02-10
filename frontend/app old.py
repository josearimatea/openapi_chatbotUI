import streamlit as st
import requests

st.title("Chatbot OpenAPI 3GPP")

pergunta = st.chat_input("Digite sua pergunta...")

if pergunta:
    with st.chat_message("user"):
        st.write(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                response = requests.post(
                    "http://localhost:5000/chat",
                    json={"mensagem": pergunta}
                )
                resposta = response.json()["resposta"]
                st.write(resposta)
            except Exception as e:
                st.error(f"Erro: {e}")