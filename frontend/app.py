# frontend/app.py
"""
Streamlit frontend for the 3GPP Chatbot RAG.
Connects to FastAPI backend with real-time streaming via SSE.
"""

import json
import requests
import streamlit as st

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="3GPP Chatbot RAG",
    page_icon="📡",
    layout="centered",
)

BACKEND_URL = "http://localhost:5000"

# ── Fetch backend info (cached per session) ──────────────────
if "backend_info" not in st.session_state:
    try:
        resp = requests.get(f"{BACKEND_URL}/info", timeout=5)
        resp.raise_for_status()
        st.session_state.backend_info = resp.json()
    except Exception:
        st.session_state.backend_info = None

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("3GPP Chatbot RAG")
    st.caption("Ask questions about 3GPP technical specifications.")

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    info = st.session_state.backend_info
    if info:
        st.markdown(
            f"**LLM:** {info['model']}  \n"
            f"**Orchestration:** {info['orchestration']}  \n"
            f"**Vector DB:** {info['vector_db']}  \n"
            f"**Embeddings:** {info['embedding_model']}  \n"
            f"**Collection:** {info['collection_name']}  \n"
            "**Data:** 3GPP TSpec-LLM"
        )
    else:
        st.markdown("⚠️ Could not load backend info.")
        if st.button("Retry", key="retry_info"):
            del st.session_state.backend_info
            st.rerun()

# ── Chat history ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📄 Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.markdown(
                        f"- **Spec:** {src.get('spec', 'n/a')} | "
                        f"**Release:** {src.get('release', 'n/a')} | "
                        f"**Series:** {src.get('series', 'n/a')} | "
                        f"**Chunk:** {src.get('chunk_index', 'n/a')}"
                    )


# ── Helper: stream tokens from SSE endpoint ──────────────────
def stream_response(question: str):
    """
    Connects to /chat/stream (SSE) and yields tokens as they arrive.
    Returns sources separately via st.session_state.
    """
    st.session_state._pending_sources = []

    try:
        with requests.post(
            f"{BACKEND_URL}/chat/stream",
            json={"message": question},
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])  # strip "data: " prefix

                if event["type"] == "token":
                    yield event["data"]
                elif event["type"] == "sources":
                    st.session_state._pending_sources = event["data"]
                elif event["type"] == "error":
                    yield f"\n\n⚠️ Error: {event['data']}"

    except requests.exceptions.ConnectionError:
        yield "⚠️ Could not connect to the backend. Is it running?"
    except requests.exceptions.Timeout:
        yield "⚠️ Request timed out."
    except Exception as e:
        yield f"⚠️ Unexpected error: {e}"


# ── User input ───────────────────────────────────────────────
question = st.chat_input("Ask about 3GPP specifications...")

if question:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Stream assistant response
    with st.chat_message("assistant"):
        response_text = st.write_stream(stream_response(question))

        # Show sources if returned
        sources = st.session_state.get("_pending_sources", [])
        if sources:
            with st.expander(f"📄 Sources ({len(sources)})"):
                for src in sources:
                    st.markdown(
                        f"- **Spec:** {src.get('spec', 'n/a')} | "
                        f"**Release:** {src.get('release', 'n/a')} | "
                        f"**Series:** {src.get('series', 'n/a')} | "
                        f"**Chunk:** {src.get('chunk_index', 'n/a')}"
                    )

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "sources": sources,
    })
