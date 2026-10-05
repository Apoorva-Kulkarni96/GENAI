from sentence_transformers import CrossEncoder

model = CrossEncoder("BAAI/bge-reranker-base")


query = "Why does overlap help when chunking documents?"
document1 = "Chunking divides a document into smaller units that can be indexed and retrieved. A fixed-size strategy splits text using a predetermined character or token size. For example, a system might create chunks of 500 tokens with 50 tokens of overlap."
document2 = "Overlap helps preserve information around chunk boundaries. If a sentence or fact falls near the end of one chunk, some of that information can appear in the next chunk as well. However, excessive overlap creates redundant chunks, increases storage, and can cause duplicate context during retrieval."
document3 = "Cosine similarity is commonly used to compare embedding vectors. A higher cosine similarity generally indicates that two vectors point in more similar directions. Vector retrieval is therefore often described as semantic search because it can retrieve text with related meaning even when the exact query words are not present."
document4 = "Vector retrieval is fast and is useful for finding a candidate set, but its similarity ordering is not always the best final relevance ordering. A reranker can take the query and each retrieved candidate together and produce a more detailed relevance score."

documents = [document1, document2, document3, document4]
pairs =[[query, doc] for doc in documents] 

scores = model.predict(pairs)


for i , (doc, score) in enumerate(zip(documents, scores), start = 1):
    print(f"\nDocument {i}")
    print(f"Score: {score:.2f}")
    print(f"Text: {doc}")


