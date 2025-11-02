import streamlit as st
import requests

# === Configuration ===
API_URL = st.secrets.get("API_URL", "http://localhost:8000/chat")

st.set_page_config(page_title="🧠 Domain RAG Chatbot", layout="wide")

# === App Title ===
st.title("🧠 Domain-Specific RAG Chatbot — Meta Llama 3.3 70B Instruct")
st.caption("Ask intelligent, context-aware questions about your custom documents!")

# === Sidebar Settings ===
with st.sidebar:
    st.header("⚙️ Settings")
    user_id = st.text_input("User ID", value="demo_user")
    top_k = st.slider("Top-K Results", 1, 10, 5)
    st.markdown("---")
    st.info("✅ Make sure your FastAPI backend is running on port 8000 before chatting.")
    st.caption("Start backend with: `uvicorn app.main:app --reload --port 8000`")

# === Initialize chat history ===
if "history" not in st.session_state:
    st.session_state.history = []

# === User Input ===
query = st.chat_input("💬 Ask something about your documents...")

# === Handle Query ===
if query:
    payload = {"user_id": user_id, "query": query}
    with st.spinner("Thinking... 🤔"):
        try:
            response = requests.post(API_URL, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            answer = data.get("answer", "No response from model.")
            sources = data.get("sources", [])
        except Exception as e:
            answer = f"⚠️ Error: {e}"
            sources = []

    # Save to chat history
    st.session_state.history.append(
        {"user": query, "assistant": answer, "sources": sources}
    )

# === Display Chat History ===
for chat in reversed(st.session_state.history):
    st.markdown(f"**🧍 You:** {chat['user']}")
    st.markdown(f"**🤖 Assistant:** {chat['assistant']}")
    if chat["sources"]:
        with st.expander("📚 Sources"):
            for s in chat["sources"]:
                st.write(f"- Doc ID: {s['doc_id']} (score: {s['score']:.4f})")
    st.markdown("---")
