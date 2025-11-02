import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB_DIR = BASE_DIR / "app" / "vector_db"
DOCS_DIR = BASE_DIR / "sample_docs"

# FAISS index file path
FAISS_INDEX_PATH = VECTOR_DB_DIR / "domain_index.faiss"

# ✅ Correct URL, loaded from environment
LLAMA_API_URL = os.getenv("LLAMA_API_URL", "https://openrouter.ai/api/v1/chat/completions")

# Embedding model
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
