import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import VECTOR_DB_DIR


class Retriever:
    def __init__(self):
        """Load SentenceTransformer model and FAISS index"""
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        index_path = VECTOR_DB_DIR / "domain_index.faiss"
        texts_path = VECTOR_DB_DIR / "domain_texts.pkl"

        # --- Load FAISS index ---
        self.index = faiss.read_index(str(index_path))

        # --- Load stored texts ---
        with open(texts_path, "rb") as f:
            self.texts = pickle.load(f)

        print(f"✅ Loaded {len(self.texts)} text chunks from {texts_path}")

        # --- Safety check for mismatches ---
        if self.index.ntotal != len(self.texts):
            print(f"⚠️ FAISS index and text count mismatch — consider rebuilding.")
            print(f"   FAISS vectors: {self.index.ntotal}, Texts: {len(self.texts)}")

    def retrieve(self, query: str, top_k: int = 3):
        """Retrieve top_k most similar text chunks for a query"""
        if not self.texts:
            print("⚠️ No texts loaded. Run ingestion first.")
            return []

        # Auto-limit top_k to available documents
        top_k = min(top_k, len(self.texts))

        # --- Encode query into embedding ---
        query_vector = self.model.encode([query]).astype("float32")

        # --- Perform FAISS similarity search ---
        D, I = self.index.search(query_vector, top_k)

        results = []
        for j, i in enumerate(I[0]):
            if 0 <= i < len(self.texts):  # ✅ Only valid indices
                results.append({
                    "doc_id": int(i),
                    "text": self.texts[i],
                    "score": float(D[0][j])
                })
            else:
                print(f"⚠️ Skipped invalid FAISS index {i}")

        # --- Sort by best (lowest) distance score ---
        results.sort(key=lambda x: x["score"])

        print(f"📚 Retrieved {len(results)} valid results for query: '{query}'")
        return results
