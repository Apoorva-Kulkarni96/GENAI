


def build_prompt(query, context):
    prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt