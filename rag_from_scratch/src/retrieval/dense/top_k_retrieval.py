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

if __name__ == "__main__":

    documents = [
        "FAISS is a vector search library",
        "HNSW is a graph algorithm",
        "Product Quantization compresses vectors"
    ]

    embeddings = [
        [0.1, 0.4, 0.8],
        [0.9, 0.2, 0.1],
        [0.2, 0.5, 0.7]
    ]
    query_embedding = [0.2, 0.5, 0.8]
    result = top_k_retrieval(
        query_embedding,
        embeddings,
        documents,
        k=2
    )

    print(result)