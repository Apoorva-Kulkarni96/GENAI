from openai import OpenAI

class Generator:
    def __init__(self, model):
        self.client = OpenAI(base_url="http://localhost:11434/v1",api_key="ollama")
        self.model = model

    def generator(self, prompt):
        response = self.client.responses.create(
            model = self.model,
            input = prompt
        )
        return response.output_text
