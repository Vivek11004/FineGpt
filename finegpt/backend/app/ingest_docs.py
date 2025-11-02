from app.config import DOCS_DIR, VECTOR_DB_DIR
from app.vector_db.faiss_utils import create_faiss_index
import os

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    # Add a sample document if none exist
    if not any(fname.endswith(".txt") for fname in os.listdir(DOCS_DIR)):
        sample_path = os.path.join(DOCS_DIR, "sample_doc.txt")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write("This is a sample document for testing FAISS ingestion.\n")
        print(f"📝 Created sample document at: {sample_path}")

    print("🚀 Starting document ingestion...")
    create_faiss_index(DOCS_DIR, VECTOR_DB_DIR)
    print("✅ Document ingestion complete! FAISS index ready.")
