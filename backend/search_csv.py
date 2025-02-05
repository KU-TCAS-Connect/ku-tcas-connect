import openai
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct
from FlagEmbedding import BGEM3FlagModel

from datetime import datetime
import pandas as pd
import ast

from services.llm_synthesizer import Synthesizer
from database.connectdb import VectorStore
from services.bge_embedding import FlagModel
from services.question_extraction import QuestionExtraction, QuestionExtractionResponse
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
            return_sparse=True, 
            return_colbert_vecs=False
        )

        dense_vector = torch.tensor(output_1['dense_vecs'][0], device=device)  # Move to GPU

        return dense_vector.cpu().numpy()  # Convert back to NumPy before sending to Qdrant
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def create_dataframe_from_results(results) -> pd.DataFrame:
    data = []
    for result in results.points:
        row = {
            "id": result.id,
            "score": result.score,
            "admission_program": result.payload.get("admission_program", ""),
            "content": result.payload.get("contents", ""),
            "reference": result.payload.get("reference", ""),
        }
        data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Convert id to string for better readability
    df["id"] = df["id"].astype(str)

    # Display the DataFrame
    return df

#################### Main ####################
chat_history = []  # Initialize chat history

query = "วิศวเครื่องกล รอบ 1 ภาคภาษอังกฤษ นานาชาติ มีเกณฑ์ยังไงบ้าง"
query_indices, query_values = compute_sparse_vector(query)

search_result = client.query_points(
    collection_name=vector_class.col_setting.collection_name["csv"],
    prefetch=[
        models.Prefetch(
            query=models.SparseVector(indices=query_indices, values=query_values),
            using="keywords",
            limit=1,
        ),
        models.Prefetch(
            query=generate_bge_embedding(query),  # <-- dense vector using BGE model
            using="",
            limit=1,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
)

#################### Print the search results (Retrieve Document) ####################
print("#################### Print the search results (Retrieve Document) ####################")
for result in search_result.points:
    print(f"Score: {result.score}")
    print(f"""{result.payload["admission_program"]}\n{result.payload["contents"]}\n{result.payload["reference"]}""")
    print("---------------------------------")

################### QuestionExtraction ####################
print("################### QuestionExtraction ####################")
thought_process, major, round_, program, program_type = QuestionExtraction.extract(query, QuestionExtractionResponse)
print(f"Extract from User Question using LLM Question Checker")
print(thought_process)
print(f"Major: {major}")
print(f"Round: {round_}")
print(f"Program: {program}")
print(f"Program Type: {program_type}")

#################### Generate Answer by LLM ####################
print("#################### Generate Answer by LLM ####################")
print("First Question")
# First question
response1 = Synthesizer.generate_response(
    question="วิศวเครื่องกล อินเตอร์ มีเกณฑ์ยังไงบ้าง", 
    context=create_dataframe_from_results(search_result), 
    history=chat_history
)
print("Answer First Question", response1.answer)

# Second question (same chat session, keeps context)
print("Second Question")
response2 = Synthesizer.generate_response(
    question="แล้วค่าเทอมเท่าไหร่?", 
    context=create_dataframe_from_results(search_result), 
    history=chat_history  # Keeps previous messages
)
print("Answer Secodn Question", response2.answer)
