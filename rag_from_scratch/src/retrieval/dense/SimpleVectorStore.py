
import numpy as np
from embeddings.cosine_similarity import cosine_similarity

class SimpleVectorStore:

    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.ids = []
        self.metadata = []

    def _top_k_retrieval(self, query_embedding, k):
        scores = []
        result = []
        for i, embedding in enumerate(self.embeddings):
            
            score = cosine_similarity(query_embedding, embedding)
            scores.append(score)

        top_indices = np.argsort(scores)[::-1][:k]
        for idx in top_indices:
            result.append(
                {    
                    "id" : self.ids[idx],
                    "document" : self.documents[idx],
                    "metadata" : self.metadata[idx],
                    "score" : scores[idx],
                            
                }
                    
            )
            
        return result

    def add(self, ids, document, embedding, metadata):
        self.ids.append(ids)
        self.documents.append(document)
        self.embeddings.append(embedding)
        self.metadata.append(metadata)

    def query(self, query_embedding, k):
        return self._top_k_retrieval(query_embedding, k)