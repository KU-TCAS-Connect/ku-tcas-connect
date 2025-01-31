from openai import OpenAI

from config.settings import OpenAISettings

class OpenaiModel:
    def __init__(self):
        self.openai_setting = OpenAISettings()
        self.openai = OpenAI(api_key=self.openai_setting.api_kay)
        
    def generate_openai_embedding(self, text):
        try:
            response = self.openai.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None