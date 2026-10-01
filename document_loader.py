import os
from typing import List, Dict, Any

def load_document(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads text content from PDF, TXT, or DOCX files.
    Returns a list of dicts with 'text', 'source_name', and 'page_number' or 'section'.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()
    chunks_with_metadata = []

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            if content.strip():
                chunks_with_metadata.append({
                    "text": content.strip(),
                    "source_name": filename,
                    "page": 1
                })

    elif ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            for idx, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    chunks_with_metadata.append({
                        "text": text.strip(),
                        "source_name": filename,
                        "page": idx + 1
                    })
        except Exception as e:
            raise RuntimeError(f"Error reading PDF file {filename}: {str(e)}")

    elif ext in [".docx", ".doc"]:
        try:
            import docx
            doc = docx.Document(file_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            full_text = "\n".join(paragraphs)
            if full_text:
                chunks_with_metadata.append({
                    "text": full_text,
                    "source_name": filename,
                    "page": 1
                })
        except Exception as e:
            raise RuntimeError(f"Error reading DOCX file {filename}: {str(e)}")
    else:
        raise ValueError(f"Unsupported document format: {ext}")

    return chunks_with_metadata

def split_text_into_chunks(documents: List[Dict[str, Any]], chunk_size: int = 600, overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Splits document texts into smaller overlapping chunks for RAG embedding.
    """
    chunked_docs = []
    chunk_counter = 0

    for doc in documents:
        text = doc["text"]
        source_name = doc["source_name"]
        page = doc.get("page", 1)

        words = text.split()
        if len(words) <= chunk_size:
            chunked_docs.append({
                "chunk_id": f"{source_name}_chunk_{chunk_counter}",
                "text": text,
                "source_name": source_name,
                "page": page
            })
            chunk_counter += 1
        else:
            # Overlapping word sliding window
            step = chunk_size - overlap
            for i in range(0, len(words), step):
                chunk_words = words[i:i + chunk_size]
                chunk_text = " ".join(chunk_words)
                if len(chunk_words) > 20:  # Avoid tiny fragment chunks
                    chunked_docs.append({
                        "chunk_id": f"{source_name}_chunk_{chunk_counter}",
                        "text": chunk_text,
                        "source_name": source_name,
                        "page": page
                    })
                    chunk_counter += 1

    return chunked_docs
