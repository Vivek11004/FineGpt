from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.memory import Memory
from app.retriever import Retriever
from app.llama_infer import generate_response
from app.config import DOCS_DIR

app = FastAPI(title="Domain RAG Chatbot API")
memory = Memory()
retriever = Retriever()

class QueryRequest(BaseModel):
    user_id: str
    query: str

@app.post("/chat")
def chat(request: QueryRequest):
    user_id = request.user_id
    query = request.query

    print(f"🧠 New query from user {user_id}: {query}")

    try:
        # 🔍 Retrieve top documents
        top_docs = retriever.retrieve(query, top_k=3)
        print(f"📚 Retrieved {len(top_docs)} relevant documents")

        # Build context
        context = "\n\n".join([doc["text"] for doc in top_docs if "text" in doc])

        # Retrieve memory
        history = memory.get(user_id)
        history_text = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in history])

        # Build prompt
        prompt = f"{history_text}\nUser: {query}\nRelevant context:\n{context}\nAssistant:"

        # Generate LLM response
        answer = generate_response(prompt)
        memory.add(user_id, query, answer)

        # ✅ Convert numpy types to native Python types before returning
        sources = [
            {
                "doc_id": int(d["doc_id"]) if hasattr(d["doc_id"], "item") else d["doc_id"],
                "score": float(d["score"]) if hasattr(d["score"], "item") else d["score"]
            }
            for d in top_docs
        ]

        return {"answer": answer, "sources": sources}

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("❌ ERROR in /chat:", e)
        raise HTTPException(status_code=500, detail=str(e))
