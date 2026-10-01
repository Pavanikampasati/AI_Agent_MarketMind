from typing import List, Dict, Any, Optional
from rag.retriever import RAGRetriever

def search_knowledge_base(query: str, retriever: Optional[RAGRetriever] = None, top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Searches uploaded documents in the knowledge base for relevant context.
    Returns list of dicts: [{'text': str, 'source_name': str, 'page': int, 'score': float}]
    """
    if retriever is None:
        return [{
            "text": "No knowledge base retriever initialized or no documents uploaded yet.",
            "source_name": "System Notice",
            "page": 1,
            "score": 0.0
        }]

    results = retriever.retrieve_context(query, top_k=top_k)
    if not results:
        return [{
            "text": "No relevant matching content found in uploaded documents for this query.",
            "source_name": "Knowledge Base",
            "page": 1,
            "score": 0.0
        }]
    return results
