import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def create_faiss_index(docs_dir, vector_db_dir):
    """
    Creates a FAISS index and saves both the index and text chunks.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = []
    embeddings = []

    # Load all text files from docs_dir
    for file in tqdm(os.listdir(docs_dir), desc="Batches"):
        if file.endswith(".txt"):
            path = os.path.join(docs_dir, file)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    texts.append(content)
                    emb = model.encode(content)
                    embeddings.append(emb)

    if not texts:
        raise ValueError("❌ No documents found to index!")

    # Convert to FAISS index
    import numpy as np
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save FAISS index and text data
    index_path = os.path.join(vector_db_dir, "domain_index.faiss")
    texts_path = os.path.join(vector_db_dir, "domain_texts.pkl")

    faiss.write_index(index, index_path)
    with open(texts_path, "wb") as f:
        pickle.dump(texts, f)

    print(f"✅ Saved FAISS index to {index_path}")
    print(f"✅ Saved text chunks to {texts_path}")
