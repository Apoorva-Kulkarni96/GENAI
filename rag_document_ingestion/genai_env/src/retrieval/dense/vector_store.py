import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.texts = [doc["text"] for doc in documents]
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.document_vectors = self.model.encode(self.texts)
        
    def cosine_similarity(self, v1, v2):
        dot_product = np.dot(v1,v2)
        mag_v1 = np.linalg.norm(v1)
        mag_v2 = np.linalg.norm(v2)
        if mag_v1 == 0.0 or mag_v2 == 0.0:
            return 0.0
        return dot_product/(mag_v1*mag_v2)
    
    def search(self, query, top_k):
        scores = []
        result = []

        query_vector = self.model.encode(query)

        for document_vector in self.document_vectors:
            score = self.cosine_similarity(
                query_vector,
                document_vector
            )
            scores.append(score)

        top_scores = np.argsort(scores)[::-1][:top_k]
      
        for idx in top_scores:
            result.append({
                "doc_id": f"Doc_{int(idx)}",
                "document": self.documents[idx],
                "score": float(scores[idx])
            })

        return result
        