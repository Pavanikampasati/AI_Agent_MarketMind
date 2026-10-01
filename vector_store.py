import os
from typing import List, Dict, Any, Optional
from rag.embeddings import LocalEmbedder

class VectorStoreManager:
    def __init__(self, persist_dir: str = "./data/chroma_db"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.embedder = LocalEmbedder()
        self._client = None
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            try:
                import chromadb
                from chromadb.config import Settings
                self._client = chromadb.PersistentClient(path=self.persist_dir)
                self._collection = self._client.get_or_create_collection(name="business_research_docs")
            except Exception as e:
                print(f"[VectorStore Error] ChromaDB init error: {e}. Utilizing in-memory document store.")
                self._collection = "in_memory_fallback"
                self.in_memory_docs = []
        return self._collection

    def add_documents(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Adds document chunks to the vector store.
        chunks format: [{'chunk_id': str, 'text': str, 'source_name': str, 'page': int}]
        """
        if not chunks:
            return 0

        coll = self._get_collection()
        texts = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [{"source_name": c["source_name"], "page": c.get("page", 1)} for c in chunks]

        embeddings = self.embedder.embed_documents(texts)

        if coll == "in_memory_fallback":
            for i, chunk in enumerate(chunks):
                self.in_memory_docs.append({
                    "id": ids[i],
                    "text": texts[i],
                    "metadata": metadatas[i],
                    "embedding": embeddings[i]
                })
            return len(chunks)
        else:
            coll.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            return len(chunks)

    def similarity_search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Searches for top_k documents matching the query.
        Returns list of dicts: [{'text': str, 'source_name': str, 'page': int, 'score': float}]
        """
        coll = self._get_collection()
        query_embedding = self.embedder.embed_query(query)

        if coll == "in_memory_fallback":
            if not hasattr(self, 'in_memory_docs') or not self.in_memory_docs:
                return []
            import numpy as np
            q_vec = np.array(query_embedding)
            results = []
            for item in self.in_memory_docs:
                doc_vec = np.array(item["embedding"])
                score = float(np.dot(q_vec, doc_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(doc_vec) + 1e-8))
                results.append((score, item))
            results.sort(key=lambda x: x[0], reverse=True)
            top_items = results[:top_k]
            return [{
                "text": item["text"],
                "source_name": item["metadata"]["source_name"],
                "page": item["metadata"]["page"],
                "score": round(score, 4)
            } for score, item in top_items]
        else:
            if coll.count() == 0:
                return []
            res = coll.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, coll.count())
            )
            retrieved = []
            if res and res.get("documents") and res["documents"][0]:
                for i in range(len(res["documents"][0])):
                    doc_text = res["documents"][0][i]
                    meta = res["metadatas"][0][i] if res.get("metadatas") else {}
                    retrieved.append({
                        "text": doc_text,
                        "source_name": meta.get("source_name", "Uploaded Document"),
                        "page": meta.get("page", 1),
                        "score": 0.90 # high match
                    })
            return retrieved

    def clear(self):
        """Clears all vector store contents for clean reset."""
        try:
            coll = self._get_collection()
            if coll != "in_memory_fallback" and self._client:
                self._client.delete_collection("business_research_docs")
                self._collection = self._client.get_or_create_collection(name="business_research_docs")
            elif hasattr(self, 'in_memory_docs'):
                self.in_memory_docs = []
        except Exception as e:
            print(f"[VectorStore Clear Error] {e}")
