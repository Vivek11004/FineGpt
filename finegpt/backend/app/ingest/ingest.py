import os
from pathlib import Path
from app.vector_db.faiss_utils import VectorDB
from app.ingest.chunker import chunk_text
from app.config import EMBED_MODEL, DOCS_DIR, VECTOR_DB_DIR

def ingest_docs():
    docs = []
    for file in Path(DOCS_DIR).glob("*.md"):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            docs.extend(chunk_text(text))

    db = VectorDB(EMBED_MODEL)
    db.build_index(docs)

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(db.index, str(VECTOR_DB_DIR / "domain_index.faiss"))
    print(f"✅ Ingested {len(docs)} chunks into FAISS index.")

if __name__ == "__main__":
    ingest_docs()
