from openai import OpenAI


class OllamaLLM:
    
    def __init__(self):
        self.client = OpenAI(
            base_url ='http://localhost:11434/v1',
            api_key='ollama',
        )

    def generate_prompt(self, prompt:str,model):
        
        response = self.client.chat.completions.create(
            model = model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        result = response.choices[0].message.content
        return result


