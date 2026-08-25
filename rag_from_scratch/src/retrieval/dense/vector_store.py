import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.document_vectors = self.model.encode(self.documents)
        

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
                "doc_id": idx,
                "score": scores[idx]
            })

        return result
        



if __name__ == "__main__":
    documents = [
            "FAISS is a vector search library",
            "HNSW is a graph algorithm",
            "Product Quantization compresses vectors",
            "FAISS is FAST"
        ]
    
   
    query_doc = "FAISS is developed by Facebook"
    obj = VectorStore(documents)
    print(obj.search(query_doc, 3))
