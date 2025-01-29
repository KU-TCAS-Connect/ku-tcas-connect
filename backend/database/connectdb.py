import uuid
from datetime import datetime
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, Distance, VectorParams

from config.settings import DatabaseSetting, CollectionSetting

class VectorStore:
    def __init__(self):
        self.db_setting = DatabaseSetting()
        self.col_setting = CollectionSetting()
        self.qdrant_client = QdrantClient(self.db_setting.service_url)

    def get_embedding(self):
        pass

    def create_collection(self):
        pass

    def uuid_from_time(self,timestamp):
        return uuid.uuid5(uuid.NAMESPACE_DNS, timestamp.isoformat())