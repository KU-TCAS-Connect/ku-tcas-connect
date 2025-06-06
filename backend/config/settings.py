import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv(dotenv_path="./.env")

class LLMSettings:
    temperature: float = 0.3
    max_tokens: Optional[int] = 3000
    max_retries: int = 3

class OpenAISettings(LLMSettings):
    api_key: str = os.getenv("OPENAI_API_KEY")
    default_model: str = "gpt-4o"
    # default_model: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-3-small"

class DatabaseSetting:
    service_url:str = os.getenv("QDRANT_URL")

class CollectionSetting:
    collection_name:object = {"csv":"kutcas_csv_metadatas_new_chunk", "txt": "kutcas_txt_metadatas_new_chunk"}
    vector_size:int = 1024
    vector_distance:str = "cosine"