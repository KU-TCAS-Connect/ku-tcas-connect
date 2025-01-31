import openai
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct
from FlagEmbedding import BGEM3FlagModel

from datetime import datetime
import pandas as pd
import ast

from database.connectdb import VectorStore
from services.llm_bge import FlagModel

load_dotenv()

vector_class = VectorStore()
client = vector_class.qdrant_client
models = vector_class.qdrant_model

flag_class = FlagModel()
model = flag_class.bge_model

client = QdrantClient("http://localhost:6333")

# Initialize the BGEM3 model
bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

def compute_sparse_vector(text):
    sentences_1 = [text]  # Use the content of the row for encoding
    output_1 = model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=False)
    
    # Extract the lexical weights (this is your sparse representation)
    lexical_weights = output_1['lexical_weights'][0]

    # Convert the lexical weights into a dictionary (index: weight)
    sparse_vector_dict = {token: weight for token, weight in lexical_weights.items()}

    indices = list(sparse_vector_dict.keys())  # Indices of the sparse vector
    values = [float(x) for x in list(sparse_vector_dict.values())]
    sparse_vector = dict(zip(indices, values))
    return indices, values

def generate_bge_embedding(text):
    try:
        # Generate dense embedding using BGE model
        sentences_1 = [text]  # The content you want to encode
        output_1 = model.encode(sentences_1, return_dense=True, return_sparse=False, return_colbert_vecs=False)

        # Extract the dense vector (embedding)
        dense_vector = output_1['dense_vecs'][0]

        return dense_vector
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

#################### Main ####################
query = "อยากทราบเกณฑ์วิศวซอฟต์แวร์และความรู้รอบ 1"
query_indices, query_values = compute_sparse_vector(query)

search_result = client.query_points(
    collection_name=vector_class.col_setting.collection_name,
    prefetch=[
        models.Prefetch(
            query=models.SparseVector(indices=query_indices, values=query_values),
            using="keywords",
            limit=5,
        ),
        models.Prefetch(
            query=generate_bge_embedding(query),  # <-- dense vector using BGE model
            using="",
            limit=5,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
)

# Print the search results
for result in search_result.points:
    print(f"Score: {result.score}")
    print(f"""{result.payload["admission_program"]}\n{result.payload["contents"]}\n{result.payload["reference"]}""")
    print("---------------------------------")
