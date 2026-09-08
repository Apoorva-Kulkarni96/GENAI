

def combine_scores(sparse_scores, dense_scores, alpha):
    hybrid_scores = {}

    all_doc_ids = set(sparse_scores) | set(dense_scores)

    for doc_id in all_doc_ids:
        bm25_score = sparse_scores.get(doc_id, 0.0)
        dense_score = dense_scores.get(doc_id, 0.0)

        hybrid_scores[doc_id] = (
            alpha * bm25_score
            + (1 - alpha) * dense_score
        )

    return hybrid_scores

def rank_top_k(hybrid_scores, top_k):
    top_scores= sorted(
        hybrid_scores.items(),
        key = lambda x : x[1],
        reverse=True
    )[:top_k]
    return [{
        "doc_id" : doc_id,
        "score" : score 
    }for doc_id, score in top_scores]


def get_ranks(result):
    ranks = {}
    for rank, doc in enumerate(result, start = 1):
        ranks[doc["doc_id"]] = rank

    return ranks


def calculate_rrf(bm25_ranks, dense_ranks, top_k, k, documents):
    all_doc_ids = set(bm25_ranks) | set(dense_ranks) 
    rrf_score={}
    top_sorted_rrf = []
    for doc_id in all_doc_ids:
        bm25_rank = bm25_ranks.get(doc_id)
        dense_rank = dense_ranks.get(doc_id)

        bm25_score = 0 if bm25_rank is None else 1/(k+bm25_rank)
        dense_score = 0 if dense_rank is None else 1/(k+dense_rank)

        rrf_score[doc_id]= bm25_score + dense_score

    sorted_rrf = sorted(rrf_score.items(), key = lambda x : x[1], reverse=True)

    for doc, score in sorted_rrf[:top_k]:
        top_sorted_rrf.append(
            {
                "doc_id" : doc,
                "document": documents[int(doc.replace("Doc_",""))],
                "score" : score
            }
        )
    return top_sorted_rrf
    

    


