import openai
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct
from FlagEmbedding import BGEM3FlagModel

from datetime import datetime
import pandas as pd
import ast

from database.connectdb import VectorStore
from backend.services.bge_embedding import FlagModel
import torch

device = "cuda(GPU)" if torch.cuda.is_available() else "CPU"
print(f"Using device: {device}")

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
    output_1 = model.encode(
        sentences_1, 
        return_dense=True, 
        return_sparse=True, 
        return_colbert_vecs=False
    )

    lexical_weights = output_1['lexical_weights'][0]

    sparse_vector_dict = {token: weight for token, weight in lexical_weights.items()}

    indices = list(sparse_vector_dict.keys())
    values = [float(x) for x in list(sparse_vector_dict.values())]

    return indices, values

def generate_bge_embedding(text):
    try:
        sentences_1 = [text]
        
        output_1 = model.encode(
            sentences_1, 
            return_dense=True, 
            return_sparse=False, 
            return_colbert_vecs=False
        )

        dense_vector = torch.tensor(output_1['dense_vecs'][0], device=device)  # Move to GPU

        return dense_vector.cpu().numpy()  # Convert back to NumPy before sending to Qdrant
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

#################### Main ####################
query = "วิศวเครื่องกล อินเตอร์ มีเกณฑ์ยังไงบ้าง"
query_indices, query_values = compute_sparse_vector(query)

search_result = client.query_points(
    collection_name=vector_class.col_setting.collection_name["csv"],
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
