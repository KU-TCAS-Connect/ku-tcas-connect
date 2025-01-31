import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="./.env")

class LLMSettings:
    max_token: int=3

class OpenAISettings:
    api_kay: str = os.getenv("OPENAI_API_KEY")
    default_model: str = os.getenv("DEFAULT_MODEL")
    embedding_model: str = os.getenv("EMBEDDING_MODEL")

class DatabaseSetting:
    service_url:str = os.getenv("QDRANT_URL")

class CollectionSetting:
    collection_name:str = "kutcas_testinsert"
    vector_size:int = 1536
    vector_distance:str = "cosine"