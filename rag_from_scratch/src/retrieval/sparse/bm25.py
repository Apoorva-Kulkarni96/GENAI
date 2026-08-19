from inverted_index import InvertedIndex
from tokenizer import tokenize


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
    b = 0.0
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

    print(calculate_bm25(query_doc, docA, documents))
    print(calculate_bm25(query_doc, docB, documents))
    print(calculate_bm25(query_doc, documents[2], documents))
    print(calculate_bm25(query_doc, docC, documents))