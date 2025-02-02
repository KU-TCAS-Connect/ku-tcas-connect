import uuid
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams

from config.settings import DatabaseSetting, CollectionSetting

class VectorStore:
    def __init__(self):
        self.db_setting = DatabaseSetting()
        self.col_setting = CollectionSetting()
        self.qdrant_client = QdrantClient(self.db_setting.service_url)
        self.qdrant_model = models

    def get_embedding(self):
        pass

    def create_collection(self, col_name):
        vector_distance = Distance.COSINE # set default
        if self.col_setting.vector_distance == "euclid":
            vector_distance = Distance.EUCLID
        elif self.col_setting.vector_distance == "dot":
            vector_distance = Distance.DOT
        elif self.col_setting.vector_distance == "manhattan":
            vector_distance = Distance.MANHATTAN

        self.qdrant_client.create_collection(
            collection_name=col_name,
            vectors_config=VectorParams(
                size=self.col_setting.vector_size,  # Size for the dense vector (for example)
                distance=vector_distance
            ),
            sparse_vectors_config={
                "keywords": self.qdrant_model.SparseVectorParams(  # Field name for sparse vectors
                    index=self.qdrant_model.SparseIndexParams(on_disk=False)
                )
            },
        )
        print(f"Collection '{self.col_setting.collection_name}' created successfully.")


    def uuid_from_time(self, timestamp):
        return uuid.uuid5(uuid.NAMESPACE_DNS, timestamp.isoformat())