import numpy as np
from embeddings.cosine_similarity import cosine_similarity

def top_k_retrieval(query_embedding, embeddings, documents, ids, metadata, k):
    scores = []
    result = []
    for i, embedding in enumerate(embeddings):
        
        score = cosine_similarity(query_embedding, embedding)
        scores.append(score)

    top_indicies = np.argsort(scores)[::-1][:k]
    for idx in top_indicies:
        result.append[
            {    
                "id" : ids[idx],
                "docs" : documents[idx],
                "metadata" : metadata[idx],
                "scores" : scores[idx],
                        
            }
                
        ]
        
    return result
