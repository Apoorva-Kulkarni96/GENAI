from inverted_index import InvertedIndex
from tokenizer import tokenize

def build_index(documents):
    obj = InvertedIndex()
    index = obj.add(documents)
    return index

def calculate_idf(query,total_documents,index):
    words = tokenize(query)
    idf_doc = [] 

    for word in words:
        if word not in index:
            idf_doc.append(0.0)
        elif word in index:
            value = index[word]
            df = len(value)
            idf_doc.append(total_documents/df)
    return idf_doc
            
def calculate_tfbm(query, document,avgdl):
    k1 =0.7
    b = 0.5
    tfbm = []
    words = tokenize(query)
    for word in words:
        tf = 0
        terms = tokenize(document)
        doc_length = len(terms)
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

def calculate_bm25(query, document, documents, index, avgdl, total_documents):
    bm25 = 0
    tf_list = calculate_tfbm(query, document,avgdl)
    idf_list = calculate_idf(query,total_documents,index)
    for tf, idf in zip(tf_list, idf_list):
        bm25 += tf * idf

    return bm25

def get_candidates(query,index):
    candidate_id = set()

    terms = tokenize(query)
    for term in terms:
        if term in index:
            val = index[term]
            for idx in val:
                candidate_id.add(idx)
    return candidate_id



    
def retrieve(query, documents, top_k):
    index = build_index(documents)
    candidate_ids = get_candidates(query, index)
    avgdl = calculate_avgdl(documents)
    total_document = len(documents)
    scores = []

    for doc_id in candidate_ids:
        score = calculate_bm25(
            query,
            documents[doc_id],
            documents,
            index,
            avgdl, 
            total_document
        )
        scores.append((doc_id, score))
    scores.sort(key=lambda x: x[1], reverse=True)

    top_k_scores = scores[:top_k]

    result = []

    for doc_id, score in top_k_scores:
        result.append({
            "doc_id": f"Doc_{doc_id}",
            "score": score
        })

    return result
if __name__ == "__main__":
    documents = [
            "FAISS is a vector search library",
            "HNSW is a graph algorithm",
            "Product Quantization compresses vectors",
            "FAISS is FAST"
        ]
    
   
    query_doc = "FAISS is developed by Facebook"


    print(retrieve(query_doc,documents,3))
    #print(get_candidates(query_doc,documents))