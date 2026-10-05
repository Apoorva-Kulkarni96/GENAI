from rag.generator import Generator
from retrieval.dense.vector_store import VectorStore
from rag.context import build_context
from rag.prompt import build_prompt
from retrieval.hybrid.hybrid_retriever import calculate_rrf, get_ranks
from retrieval.sparse.bm25 import retrieve
from ingestion.chunking import TextChunker
from evaluation.retrieval_metrics import *
import json

class RAGPipeline:
    def __init__(self, documents, model):
        self.documents = documents
        self.model  = model
        self.obj_den = VectorStore(self.documents)
        self.obj_gen = Generator(self.model)

    def ask(self, query, top_k, mode="hybrid"):

        if mode == "dense":
            results = self.obj_den.search(query, top_k, section="3. Chunking")

        elif mode == "bm25":
            results = retrieve(query, self.documents, top_k)

        elif mode == "hybrid":
            results_dense = self.obj_den.search(query, top_k)
            results_sparse = retrieve(query, self.documents, top_k)

            sparse_ranks = get_ranks(results_sparse)
            dense_ranks = get_ranks(results_dense)

            results = calculate_rrf(
                sparse_ranks,
                dense_ranks,
                top_k,
                60,
                self.documents
            )

        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        context = build_context(results)
        prompt = build_prompt(query, context)
        response = self.obj_gen.generator(prompt)

        return response, results
    
if __name__ == "__main__":
    with open("/Users/apoor/OneDrive/Desktop/git/GENAI/rag_document_ingestion/genai_env/data/rag_corpus.txt", "r", encoding="utf-8") as f:
        text = f.read()
        chunker = TextChunker()


        chunks = chunker.recursive_chunk_with_sections(text)

    with open("/Users/apoor/OneDrive/Desktop/git/GENAI/rag_document_ingestion/genai_env/data/evaluation_queries.json", "r") as f:
        evaluation_data = json.load(f)

    top_k = 3

    documents = [
        {
            "doc_id": f"Doc_{i}",
            "text": chunk["text"],
            "metadata": {
                "source": "rag_corpus.txt",
                "doc_type": "text",
                "section": chunk["section"]
            }
        }
        for i, chunk in enumerate(chunks)
    ]

    rag = RAGPipeline(documents, "gemma:2b")

    all_recall = []
    all_precision = []
    all_rr = []

    top_k = 3

    for mode in ["dense", "bm25", "hybrid"]:

        all_results = []
        all_relevant_sections = []

        for item in evaluation_data:

            answer, results = rag.ask(
                item["question"],
                top_k,
                mode
            )

            all_results.append(results)
            all_relevant_sections.append(
                item["relevant_sections"]
            )

        recalls = []
        precisions = []
        reciprocal_ranks = []

        for results, relevant_sections in zip(
            all_results,
            all_relevant_sections
        ):

            recalls.append(
                recall_at_k(
                    results,
                    relevant_sections,
                    top_k
                )
            )

            precisions.append(
                precision_at_k(
                    results,
                    relevant_sections,
                    top_k
                )
            )

            reciprocal_ranks.append(
                reciprocal_rank(
                    results,
                    relevant_sections
                )
            )

        average_recall = sum(recalls) / len(recalls)
        average_precision = sum(precisions) / len(precisions)
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

        print("\n====================")
        print("Mode:", mode)
        print("====================")
        print(f"Recall@{top_k}:", average_recall)
        print(f"Precision@{top_k}:", average_precision)
        print("MRR:", mrr)