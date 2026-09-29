

def recall_at_k(results, relevant_sections, top_k):

    retrieved_sections = []

    for item in results[:top_k]:
        section = item["document"]["metadata"]["section"]

        if section not in retrieved_sections:
            retrieved_sections.append(section)

    relevant_retrieved_count = 0

    for section in retrieved_sections:
        if section in relevant_sections:
            relevant_retrieved_count += 1

    return relevant_retrieved_count / len(relevant_sections)

def precision_at_k(results, relevant_sections, top_k):

    retrieved_sections = []

    for item in results[:top_k]:
        section = item["document"]["metadata"]["section"]

        if section not in retrieved_sections:
            retrieved_sections.append(section)

    relevant_retrieved_count = 0

    for section in retrieved_sections:
        if section in relevant_sections:
            relevant_retrieved_count += 1

    if not retrieved_sections:
        return 0.0

    return relevant_retrieved_count / len(retrieved_sections)

def reciprocal_rank(results, relevant_sections):

    for rank, item in enumerate(results, start=1):

        section = item["document"]["metadata"]["section"]

        if section in relevant_sections:
            return 1 / rank

    return 0.0

def mean_reciprocal_rank(all_results, all_relevant_docs):

    reciprocal_ranks = []

    for results, relevant_docs in zip(all_results, all_relevant_docs):
        rank = reciprocal_rank(results, relevant_docs)
        reciprocal_ranks.append(rank)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)
