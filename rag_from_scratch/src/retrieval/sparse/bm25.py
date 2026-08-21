from inverted_index import InvertedIndex
from tokenizer import tokenize
import numpy as np

def calculate_idf(query,documents):
    words = tokenize(query)
    idf_doc = []
    total_document = len(documents)
    obj = InvertedIndex()
    index = obj.add(documents)
    for word in words:
        if word not in index:
            idf_doc.append(0.0)
        elif word in index:
            value = index[word]
            df = len(value)
            idf_doc.append(total_document/df)
    return idf_doc
            
def calculate_tfbm(query, document, documents):
    k1 =0.7
    b = 0.5
    tfbm = []
    words = tokenize(query)
    for word in words:
        tf = 0
        terms = tokenize(document)
        doc_length = len(terms)
        avgdl = calculate_avgdl(documents)
        for term in terms:
            if term == word:
                tf += 1
        tfbm.append((tf*(k1+1)/(tf+k1*(1-b+b*(doc_length/avgdl)))))
    return tfbm
            
def calculate_avgdl(documents):
    avgdl=0
    documents_len = len(documents)
    for doc in documents:
        terms = tokenize(doc)
        avgdl += len(terms)

    return avgdl/documents_len

def calculate_bm25(query, document, documents):
    bm25 = 0
    tf_list = calculate_tfbm(query, document, documents)
    idf_list = calculate_idf(query,documents)
    for tf, idf in zip(tf_list, idf_list):
        bm25 += tf * idf

    return bm25
def retrieve(query, documents, top_k):
    scores= []
    for idx, document in enumerate(documents):
        score = calculate_bm25(query, document, documents)
        scores.append(score)
    top_k_scores = np.argsort(scores)[::-1][:top_k]
    result = []

    for idx in top_k_scores:
        result.append(
            {
                "doc_id": f"Doc_{idx}",
                "score" : scores[idx]

            }
        )
    return result
if __name__ == "__main__":
    documents = [
            "FAISS is a vector search library",
            "HNSW is a graph algorithm",
            "Product Quantization compresses vectors",
            "FAISS is FAST"
        ]
    
   
    query_doc = "FAISS is developed by Facebook"
    keyword = "FAISS"
    docA = documents[0]
    docB = documents[1]
    docC = documents[3]

    print(retrieve(query_doc,documents,3))