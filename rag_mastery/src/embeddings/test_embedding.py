from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss

model = SentenceTransformer("all-MiniLM-L6-v2")

query = "Tell me about cars"

sentences = [
    "A car is a road vehicle used for transportation.",
    "An automobile has four wheels.",
    "Electric vehicles use batteries.",
    "Motorcycles have two wheels.",
    "Bananas are rich in potassium.",
    "Apples are healthy fruits.",
    "Cats are popular pets.",
    "Dogs are loyal animals.",
    "Transformers use self-attention.",
    "RAG combines retrieval with generation.",
]

embeddings = model.encode(sentences)
print(embeddings.dtype)
query_embeddings = model.encode(query)
query_embeddings = query_embeddings.reshape(1,-1)


score = cosine_similarity(query_embeddings, embeddings)




indices = np.argsort(score[0])[::-1]

for i in range(3):
    print("Score :", score[0][indices][i])
    print("Sentence :", sentences[indices[i]])


