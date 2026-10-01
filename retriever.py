import os
from typing import List, Dict, Any
from rag.document_loader import load_document, split_text_into_chunks
from rag.vector_store import VectorStoreManager

class RAGRetriever:
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

    def process_and_index_file(self, file_path: str) -> Dict[str, Any]:
        """
        Loads, chunks, and indexes a single uploaded document.
        """
        raw_docs = load_document(file_path)
        chunks = split_text_into_chunks(raw_docs)
        count = self.vector_store.add_documents(chunks)
        return {
            "status": "success",
            "file_name": os.path.basename(file_path),
            "chunks_indexed": count
        }

    def retrieve_context(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieves relevant context chunks for a query from the knowledge base.
        """
        return self.vector_store.similarity_search(query, top_k=top_k)
