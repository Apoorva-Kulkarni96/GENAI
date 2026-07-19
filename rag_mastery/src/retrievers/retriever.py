import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.document = None
        self.embeddings = None
        self.index = None

    def fit(self, document:str):
        self.document = document
        self.embeddings = self.embedding_model.encode(self.document)
        #self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index = faiss.IndexHNSWFlat(self.embeddings.shape[1])
        self.index.add(self.embeddings)

        
    def retrieve(self, query:str, k:int):

        if self.embeddings is None or self.document is None:
            raise ValueError("Retreiver is not fitted. call fit function first")
        query_embeddings = self.embedding_model.encode(query)
        query_embeddings = query_embeddings.reshape(1, -1)
        distance, indices = self.index.search(query_embeddings, k)
        results = []
        for idx, dist in zip(indices[0], distance[0]):
            results.append(
                { 
                    "score" : float(dist),
                    "sentence" : self.document[idx]
                }
            )
        return results
        """
        scores = cosine_similarity(query_embeddings, self.embeddings)

        top_scores = np.argsort(scores[0])[::-1][:k]
        results = []
        for idx in top_scores:
            results.append(
                {
                    "Score" : float(scores[0][idx]),
                    "Sentences" : self.document[idx]
                }
            )
        return results
        """
        