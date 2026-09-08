
def normalization(scores):
    score = []
    x_min = min(scores.values())
    x_max = max(scores.values())
    if x_min == x_max:
        return {doc_id : 0.0 for doc_id in scores}
    normalized_score = {}
    for doc_id, score in scores.items():
        normalized_score[doc_id] = (score - x_min)/(x_max - x_min)
    return normalized_score
