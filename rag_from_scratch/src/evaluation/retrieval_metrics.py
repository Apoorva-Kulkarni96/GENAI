from rag.rag_pipeline import RAGPipeline

def recall_at_k(results,relevant_docs, top_k):
    retrieved_docs = []
    relevant_retrieved_count = 0
    for item in results[:top_k]:
        retrieved_docs.append(item['doc_id'])
    for doc in retrieved_docs:
        if doc in relevant_docs:
            relevant_retrieved_count += 1
    return  relevant_retrieved_count/len(relevant_docs)

def precision_at_k(results, relevant_docs, top_k):
    retrieved_docs = []
    relevant_retrieved_count = 0
    for item in results[:top_k]:
        retrieved_docs.append(item['doc_id'])
    for doc in retrieved_docs:
        if doc in relevant_docs:
            relevant_retrieved_count += 1
    return  relevant_retrieved_count/len(retrieved_docs)


def reciprocal_rank(results, relevant_docs):
    for rank, item in enumerate(results, start =1):
        if item['doc_id'] in relevant_docs:
            return 1/rank
    return 0.0


def mean_reciprocal_rank(all_results, all_relevant_docs):

    reciprocal_ranks = []

    for results, relevant_docs in zip(all_results, all_relevant_docs):
        rank = reciprocal_rank(results, relevant_docs)
        reciprocal_ranks.append(rank)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)

if __name__ == "__main__":
    documents = [
        "FAISS is a vector search library",
        "HNSW is a graph algorithm",
        "Product Quantization compresses vectors",
        "FAISS is FAST"
    ]

    top_k = 3

    rag = RAGPipeline(documents, "gemma:2b")

    evaluation_data = [
        {
            "query": "What is FAISS?",
            "relevant_docs": ["Doc_0", "Doc_3"]
        },
        {
            "query": "What is HNSW?",
            "relevant_docs": ["Doc_1"]
        },
        {
            "query": "What is Product Quantization?",
            "relevant_docs": ["Doc_2"]
        }
    ]

    all_results = []
    all_relevant_docs = []
    avg_recall = []
    avg_precision = []

    rr_values = []
    avg_recall = []
    avg_precision = []

    for item in evaluation_data:

        answer, results = rag.ask(item["query"], top_k)

        recall = recall_at_k(
            results,
            item["relevant_docs"],
            top_k
        )

        precision = precision_at_k(
            results,
            item["relevant_docs"],
            top_k
        )

        rr = reciprocal_rank(
            results,
            item["relevant_docs"]
        )

        avg_recall.append(recall)
        avg_precision.append(precision)
        rr_values.append(rr)

        print("Query:", item["query"])
        print(f"Recall@{top_k}: {recall}")
        print(f"Precision@{top_k}: {precision}")
        print("RR:", rr)
        print()

    average_recall = sum(avg_recall) / len(avg_recall)
    average_precision = sum(avg_precision) / len(avg_precision)
    mrr = sum(rr_values) / len(rr_values)

    print(f"Average Recall@{top_k}:", average_recall)
    print(f"Average Precision@{top_k}:", average_precision)
    print("MRR:", mrr)