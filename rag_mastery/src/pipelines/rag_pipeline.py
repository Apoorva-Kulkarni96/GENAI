


class RAGPipeline:

    def __init__(self, retriever, prompt_builder, llm):
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm = llm
       


    def ask(self, query:str,k:int, model):
        results = self.retriever.retrieve(query,k)
        prompt = self.prompt_builder.build(query,results)
        final_results = self.llm.generate_prompt(prompt, model)

        return final_results


        