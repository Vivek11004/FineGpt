from typing import List
import re

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Splits a long text into overlapping chunks for embedding and retrieval.
    """
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # overlap between chunks
    
    return chunks
