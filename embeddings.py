from typing import List
import numpy as np

class LocalEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._fallback = False

    def _init_model(self):
        if self._model is None and not self._fallback:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"[RAG Embeddings Warning] Could not load sentence-transformers model ({e}). Using lightweight TF-IDF embedding fallback.")
                self._fallback = True

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self._init_model()
        if self._model and not self._fallback:
            try:
                embeddings = self._model.encode(texts, convert_to_numpy=True)
                return embeddings.tolist()
            except Exception as e:
                print(f"[RAG Embeddings Error] {e}. Switching to fallback.")
                self._fallback = True

        # Fallback embedding generator (TF-IDF vectorizer representation normalized to 384 dim)
        return self._tf_idf_fallback(texts)

    def embed_query(self, query: str) -> List[float]:
        return self.embed_documents([query])[0]

    def _tf_idf_fallback(self, texts: List[str]) -> List[List[float]]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=384)
        try:
            matrix = vectorizer.fit_transform(texts).toarray()
            # Pad to 384 dimensions if fewer features found
            if matrix.shape[1] < 384:
                padding = np.zeros((matrix.shape[0], 384 - matrix.shape[1]))
                matrix = np.hstack([matrix, padding])
            return matrix.tolist()
        except Exception:
            # Absolute simple bag-of-words fallback hash
            res = []
            for t in texts:
                vec = [0.0] * 384
                for word in t.lower().split():
                    idx = abs(hash(word)) % 384
                    vec[idx] += 1.0
                norm = np.linalg.norm(vec) or 1.0
                res.append((np.array(vec) / norm).tolist())
            return res
