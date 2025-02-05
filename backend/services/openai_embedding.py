import logging
from typing import List
from openai import OpenAI
from config.settings import OpenAISettings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIEmbeddingGenerator:

    def __init__(self):
        self.settings = OpenAISettings()
        self.client = OpenAI(api_key=self.settings.api_key)

    def generate_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-small",
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
