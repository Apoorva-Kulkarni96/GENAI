

class PromptBuilder:
    def __init__(self):

        pass

    def build(self, query:str, results:list[dict]):
        context = "\n\n".join(item["sentence"] for item in results)

        prompt = f"""
        You are a helpful AI assistant.

        Your task is to answer the user's question **only** using the provided context.

        Instructions:

        * Read the provided context carefully before answering.
        * Use only the information available in the context.
        * Do not use your own background knowledge or make assumptions.
        * If the answer is not present in the context, respond with:
        **"I don't know based on the provided context."**
        * If multiple context passages contain conflicting information, clearly mention the conflict instead of choosing one arbitrarily.
        * Keep the answer concise, accurate, and directly relevant to the user's question.

        Context:
        {context}

        Question:
        {query}

        Answer:


        """

        return prompt