import logging
from typing import Any, Dict, List, Type

import instructor
from openai import OpenAI
from pydantic import BaseModel

from config.settings import OpenAISettings, get_settings


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LLMFactory:

    def __init__(self, provider: str):
        self.provider = provider
        self.settings = OpenAISettings()
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        client_initializers = {
            "openai": lambda s: instructor.from_openai(OpenAI(api_key=s.api_key)),
        }

        initializer = client_initializers.get(self.provider)
        if initializer and self.settings:
            return initializer(self.settings)
        
        logger.error(f"Unsupported LLM provider: {self.provider}")
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Any:
        completion_params = {
            "model": kwargs.get("model", self.settings.default_model),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "max_tokens": kwargs.get("max_tokens", self.settings.max_tokens),
            "response_model": response_model,
            "messages": messages,
        }

        return self.client.chat.completions.create(**completion_params)