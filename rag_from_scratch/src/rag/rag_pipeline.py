from rag.generator import Generator
from retrieval.dense.vector_store import VectorStore
from rag.context import build_context
from rag.prompt import build_prompt
from retrieval.hybrid.hybrid_retriever import calculate_rrf, get_ranks
from retrieval.sparse.bm25 import retrieve


class RAGPipeline:
    def __init__(self, documents, model):
        self.documents = documents
        self.model  = model
        self.obj_den = VectorStore(self.documents)
        self.obj_gen = Generator(self.model)

    def ask(self, query, top_k):
        
        results_dense = self.obj_den.search(query, top_k)
        results_sparse = retrieve(query, self.documents, top_k)
        sparse_ranks = get_ranks(results_sparse)
        dense_ranks = get_ranks(results_dense)
        results = calculate_rrf(sparse_ranks, dense_ranks, top_k, 60, self.documents)
        context = build_context(results)
        prompt = build_prompt(query, context)
        response = self.obj_gen.generator(prompt)
        return response
    
if __name__ == "__main__":
    documents = [
            "FAISS is a vector search library",
            "HNSW is a graph algorithm",
            "Product Quantization compresses vectors",
            "FAISS is FAST"
        ]
    
   
    #query_doc = "FAISS is developed by Facebook"
    query_doc = "What is FAISS?"
    rag = RAGPipeline(documents, "gemma:2b")
    answer = rag.ask(query_doc, 3)
    print(answer)


