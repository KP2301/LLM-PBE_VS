import os
from together import Together
from models.LLMBase import LLMBase

class TogetherModels(LLMBase):
    def __init__(self, api_key=None, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", max_tokens=256, temperature=0.7):
        self.model_name_hf = model
        self.tokenizer = None

        super().__init__(api_key=api_key)

        if api_key:
            self.client = Together(api_key=api_key)
        elif "TOGETHER_API_KEY" in os.environ:
            self.client = Together(api_key=os.environ["TOGETHER_API_KEY"])
        else:
            self.client = Together()

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def load_model(self):
        pass

    def query(self, prompt):
        return self.query_remote_model(prompt)

    def query_remote_model(self, prompt_or_messages):
        try:
            if isinstance(prompt_or_messages, list):
                messages = prompt_or_messages
            else:
                messages = [{"role": "user", "content": str(prompt_or_messages)}]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Together AI API Error: {e}")
            return ""