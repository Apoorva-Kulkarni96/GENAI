from rag.rag_pipeline import RAGPipeline

#def extract_claims(answer):




if __name__ == "__main__":
    
    documents = [
            "FAISS is a vector search library",
            "HNSW is a graph algorithm",
            "Product Quantization compresses vectors",
            "FAISS is FAST"
        ]
    
    top_k = 3
    query = "What is FAISS?"

    rag = RAGPipeline(documents, "gemma:2b")
    answer, results = rag.ask(query, top_k)

    for i in results:
        print(i['document'])